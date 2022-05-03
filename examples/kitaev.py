#!/bin/env python
#
# 11/03/19
# Chris Self
#
import sys,time,json
import numpy as np
from mpi4py import MPI
# import ptmpi packages
import ptmpi
from ptmpi import swaphandler as PtMPI
from ptmpi import filehandler as PtMPI_out

import numpy as np

from koala.example_graphs import generate_hex_square_oct,generate_honeycomb,generate_tri_non
from koala.graph_color import color_lattice
from koala.pointsets import generate_bluenoise
from koala.voronization import generate_lattice
from koala.graph_color import color_lattice
from koala.graph_utils import plaquette_spanning_tree
from koala.plotting import plot_lattice, peru_friendly_colour_scheme
from koala.flux_finder import find_flux_sector, fluxes_from_bonds, fluxes_to_labels
from koala.lattice import Lattice
import koala.plotting as pl

from scipy.stats import sem

def construct_Ajk(lattice: Lattice, edge_colouring: np.ndarray, ujk: np.ndarray, J_values: np.ndarray, K:np.float32 =0.2):
    """construct the A matrix that the hamiltonian uses
    :param lattice: system to construct the matrix on
    :type lattice: Lattice
    :param edge_colouring: colouring for the edges must be a set of ±1 values
    :type edge_colouring: np.ndarray
    :param ujk: bond signs
    :type ujk: np.ndarray
    :param J_values: j values
    :type J_values: np.ndarray
    :param K: three spin field strength
    :type K: np.float32
    :return: the A matrix
    :rtype: np.ndarray
    """
    edge_type_list = J_values[edge_colouring]
    bond_values = 2*edge_type_list*ujk

    ham = np.zeros((lattice.n_vertices, lattice.n_vertices))
    ham[lattice.edges.indices[:,1], lattice.edges.indices[:,0]] = bond_values
    ham[lattice.edges.indices[:,0], lattice.edges.indices[:,1]] = -bond_values

    K_ham = np.zeros((lattice.n_vertices, lattice.n_vertices))
    K_ham[lattice.edges.indices[:,1], lattice.edges.indices[:,0]] = ujk
    K_ham[lattice.edges.indices[:,0], lattice.edges.indices[:,1]] = -ujk
    K_ham = K_ham @ K_ham
    K_ham *= 2*K
    K_ham = np.tril(-1*K_ham) + np.triu(K_ham)
    np.fill_diagonal(K_ham, 0)

    return 1.0j*(ham - K_ham)

def flux_sector_thermal_average_energy(energies, T):
    return -np.sum((energies / 2)*np.tanh(energies / (2*T)))

def flux_sector_thermal_average_energy_dbeta(energies, T):
    return -np.sum((energies / 2)**2 * (1 / np.cosh(energies / (2*T)))**2)

def random_flip(lattice, ujk_base):
    fluxes = fluxes_from_bonds(lattice, ujk_base)
    s1, s2 = np.random.choice(np.arange(lattice.n_plaquettes), (2,), replace=False) # flip 2 plaquettes
    fluxes[s1] *= -1
    fluxes[s2] *= -1
    ujk_out = find_flux_sector(lattice, fluxes)
    return ujk_out

def make_decision_fn(lattice, colouring):
  def kitaev_pt_decision_function(args_1,args_2):
      """
      return decision to accept or reject swap
      accept swap with probability min(1,e^{-E_1 beta_2 - E_2 beta_1}/e^{-E_1 beta_1 - E_2 beta_2})
      """
      # unpack args
      E_1,beta_1 = args_1[:2]
      fermion_energies_1 = np.array(args_1[2:])
      E_2,beta_2 = args_2[:2]
      fermion_energies_2 = np.array(args_2[2:])

      E_1_prime = flux_sector_thermal_average_energy(fermion_energies_1, 1 / beta_2)
      E_2_prime = flux_sector_thermal_average_energy(fermion_energies_2, 1 / beta_1)
      # compute swap probability
      prob_switch = min( 1, np.exp( (-E_1_prime*beta_2 - E_2_prime*beta_1 + E_1*beta_1 + E_2*beta_2) ) )
      # return decision
      return ( np.random.random()<prob_switch )
  return kitaev_pt_decision_function

if __name__ == '__main__':
    print("Program running")
    # --------
    # setup
    # --------

    # system parameters:
    L = 7 # system dimension
    J_values = np.array([1., 1., 1.]) # spin-spin coupling
    K = 0.1 # external magnetic field coupling

    # output file name
    output_name = 'kitaev'

    # fix total number of pt swap rounds
    number_swaps = 1000

    # each process has a copy of the same temperature set
    T_logset = np.concatenate((
      np.linspace(-2,0,192),
      np.linspace(0,2,64)
    ))
    betas = np.array([ 1.*10**(-bb) for bb in T_logset ])

    # -----------------
    # initialisation
    # -----------------

    # initialise the MPI evironment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if not len(betas)==comm.Get_size():
        print('Error, the number of temperatures is different than the number of chains.\n'\
            +'Aborting simulation.', flush=True)
        exit()
    # announce start
    if rank==(comm.Get_size()-1):
        print(sys.argv[0]+' initialising...\n'+'-'*40, flush=True)
    comm.Barrier()
    begin_time = time.time()

    # initialise this process's copy of the system
    print(str(rank)+': initialising copy of the system')
    #points = np.random.uniform(size=(224,2))
    #lattice = generate_lattice(points)
    lattice, colouring = generate_honeycomb(L, return_colouring=True)
    spanning_tree = plaquette_spanning_tree(lattice)
    #lattice, colouring = lattice, color_lattice(lattice)
    ujk = find_flux_sector(lattice)

    # compute starting energy
    ham = construct_Ajk(lattice, colouring, ujk, J_values, K)
    energies = np.abs((np.sort(np.linalg.eigvalsh(ham)))[:ham.shape[0]//2])
    E = flux_sector_thermal_average_energy(energies, 1 / betas[rank])

    # initialise ptmpi controller object
    decision_func = make_decision_fn(lattice, colouring)
    mpi_pt_handler = PtMPI.swaphandler( comm,rank,number_swaps=number_swaps,verbose=True,decision_func=decision_func)

    # -------------------
    # mcmc and pt loop
    # -------------------

    # open output file array
    if rank==(comm.Get_size()-1):
        print('initalising shared output files...')
    with PtMPI_out.filehandler(comm,filename='output/timeseries'+output_name) as out_file:
        # wrap an array in output file
        with out_file.wrap_array():

            # store start time
            start_time = time.time()
            start_block_time = time.time()
            vortex_density = 0.0
            de_dbeta = 0.0

            # parallel tempering swaps are the unit of our monte-carlo time and each unit
            # of time corresponds to a Metropolis sweep i.e. O(L^2) metropolis steps
            for swaps_counter in range(number_swaps):
                # print progress update every X bins
                progress_time_unit = max(10,int(np.floor(number_swaps/100.)))
                make_noise = ((rank==(comm.Get_size()-1)) and (swaps_counter%progress_time_unit==0) and (swaps_counter>0))
                if make_noise:
                    end_block_time = time.time()
                    print('-'*15)
                    print('process 0 at swaps_counter '+str(swaps_counter))
                    print('last block of '+str(progress_time_unit)+' bins took: '\
                          +"{0:.3g}".format(end_block_time-start_block_time)+' seconds')
                    start_block_time = time.time()
                # mcmc sweep
                # -------------
                for mc_step in range(L**2):
                    beta_index = mpi_pt_handler.get_current_temp_index()

                    new_ujk = random_flip(lattice, ujk)
                    ham = construct_Ajk(lattice, colouring, new_ujk, J_values, K)
                    energies = np.abs((np.sort(np.linalg.eigvalsh(ham)))[:ham.shape[0]//2])
                    delta_E = flux_sector_thermal_average_energy(energies, 1 / betas[beta_index]) - E

                    # accept or reject
                    acceptance_probability = min(1.,np.exp(-betas[beta_index]*delta_E))
                    if np.random.random()<acceptance_probability:
                        ujk = new_ujk
                        E += delta_E
                        vortex_density = 1.0 - sum((fluxes_from_bonds(lattice,ujk)+1.0)/2.0) / lattice.n_plaquettes
                        de_dbeta = flux_sector_thermal_average_energy_dbeta(energies, 1 / betas[beta_index])

                # output current state
                beta_index = mpi_pt_handler.get_current_temp_index()
                output_data = {"rank":rank,"beta":betas[beta_index],\
                               "energy":E,"vortex_density":vortex_density,"de_dbeta":de_dbeta}
                out_file.dump(beta_index,output_data)

                # parallel tempering swap step
                # -------------------------------
                try:
                    curr_beta_index = mpi_pt_handler.get_current_temp_index()
                    alt_beta_index = mpi_pt_handler.get_alternative_temp_index()
                    success_flag = mpi_pt_handler.pt_step( E,betas[curr_beta_index],*tuple(energies) )
                except PtMPI.NoMoreSwaps:
                    print('PtMPI attempted a swap but was at the end of pt_subsets')
                    break

    # record the run time of the stage and add the runtime to the task_spec
    run_time = (time.time()-begin_time)
    if rank==(comm.Get_size()-1):
        print('total runtime: '+str(run_time)+' seconds.')
