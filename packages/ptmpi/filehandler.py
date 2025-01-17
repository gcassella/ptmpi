#
# 24/04/2016
# Chris Self
#
import sys
import json
from mpi4py import MPI
from contextlib import contextmanager

class filehandler(object):
	"""
	This class provides a context manager to wrap an array of shared MPI output files. This allows each rank 
	to output to a file corresponding to its current temperature index, which changes through pt swaps.

	The intended use is to generate an array (a timeseries) of dictionary objects containing the important
	information about the state of the system. Therefore, in addition to the main context manager another
	context manager `wrap_array` is provided. 
	"""

	def __init__(self,mpi_comm_env,filename='ptmpi',label='T',amode=MPI.MODE_WRONLY|MPI.MODE_CREATE):
		""" """
		self.mpi_comm = mpi_comm_env
		self.fbasename = filename
		self.flabel = label
		self.amode = amode
		self.files = []
		# flag used to track if a comma is needed before the next element
		self.in_array = False
		self.array_between_elements = False

	def __enter__(self):
		"""
		context manager enter method. this is assumed to be at the start of the program where T_index==mpi_rank.
		each process opens a shared MPI file for their current temperature.
		"""
		num_proc = self.mpi_comm.Get_size()
		for rank in range(num_proc):
			self.files.append( MPI.File.Open(self.mpi_comm, self.fbasename+'_'+self.flabel+str(rank)+'.json', self.amode) )
		self.mpi_comm.Barrier()
		return self

	def __exit__(self,*args):
		""" 
		context manager exit method, ensures all files are closed.
		"""
		self.mpi_comm.Barrier()
		for f in self.files:
			f.Close()

	def enter_array(self):
		""" 
		wrap_array enter method. An array is opened in all MPI files. This is delegated to the root process.
		"""
		self.in_array = True
		if self.mpi_comm.Get_rank()==0:
			for f in self.files:
				f.Write_shared([b'[\n', 2, MPI.CHAR])
		self.mpi_comm.Barrier()

	def exit_array(self):
		""" 
		wrap_array exit method. The array is closed in all MPI files. This is delegated to the root process.
		"""
		self.mpi_comm.Barrier()
		if self.mpi_comm.Get_rank()==0:
			for f in self.files:
				f.Write_shared([b'\n]', 2, MPI.CHAR])
		self.in_array = False

	@contextmanager
	def wrap_array(self):
		""" 
		context manager to wrap an output array in all MPI files.
		"""
		self.enter_array()
		try:
			yield
		finally:
			self.exit_array()

	def dump(self,fileindex,obj):
		""" 
		writes a dictionary object `obj` to a specified MPI file `fileindex`
		"""
		json_string = ''
		if self.array_between_elements:
			json_string = ',\n'
		json_string = json_string + json.dumps(str(obj),indent=2)
		self.files[fileindex].Write_shared([json_string.encode('utf-8'), len(json_string), MPI.CHAR])
		self.array_between_elements = self.in_array
