# ptmpi
A python class that coordinates an MPI implementation of parallel tempering

<!---
## Installation

Links: 
MPICH: http://mpitutorial.com/tutorials/installing-mpich2/
mpi4py: http://pythonhosted.org/mpi4py/
-->

## Usage

[Parallel tempering](https://en.wikipedia.org/wiki/Parallel_tempering) is a monte-carlo method used to obtain equilibrium statistics for a physical system over a range of temperatures. When the energy landscape of the system is complex it can hugely speed up the convergence of ensemble averages, especially at low temperatures. It works by simulating *N* copies of the system (replicas) evolving independently at different temperatures [*T1*, *T2*, *T3*, ... ]. Periodically replicas at different temperatures are exchanged with some probability.

This python class (`ptmpi.PtMPI`) supports a fully parallelised implementation of parallel tempering using mpi4py (message passing interface for python). Each replica runs as a separate parallel process and they communicate via an mpi4py object. To minimise message passing the replicas stay in place and only the temperatures are exchanged between the processes. It is this exchange of temperatures that ptmpi handles.

The class is independent of the system being simulated or any the details of the simulation, including what the temperatures [*T1*, *T2*, *T3*, ... ] are. It behaves as a black box to tell the process what its position temperature in the list of temperatures is. 

### Example code 

Python script that uses ptmpi (main.py):

```python
# import mpi4py package
from mpi4py import MPI

# import ptmpi packages
import ptmpi
from ptmpi import PtMPI

import (other packages)...

if __name__ == '__main__':
	
	# initialise the MPI evironment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ...
    # specify model parameters, initialise this replica
    # ...

    # define some set of temperatures
    temps = [ ... ]

    # decide the total number of pt-swaps that will be run
    length_of_program = ...

    # initialise ptmpi object
    pt_mpi_obj = PtMPI.swaphandler( comm,rank,number_swaps=length_of_program )

    for time_step in range(length_of_program):

    	# get the current temperature from the ptmpi object
    	curr_temp = temps[ pt_mpi_obj.get_current_temp_index() ]

    	# ...
    	# evolve the replica e.g. under metropolis, output/store data, etc.
    	# ...

    	# gather the data needed to decide the parallel tempering swap
    	curr_energy = ...
    	alt_temp = temps[ pt_mpi_obj.get_alternative_temp_index() ]

    	# parallel tempering swap step
    	mpi_pi_handler.pt_step( curr_energy, curr_temp, alt_temp )

   	# ...
   	# finalise simulation
   	# ...
```

This is then run in terminal with the command:

```bash
$ mpiexec -n 16 python main.py
```