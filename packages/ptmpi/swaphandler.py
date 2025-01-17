#
# 24/04/2016
# Chris Self
#
import sys
import json
from mpi4py import MPI
import numpy as np

class NoMoreSwaps(Exception):
    """
    swaphandler is instanced with a fixed number of available swap rounds, if this number
    is exceeded swaphandler raises this exception.
    """
    pass

class BadDecisionFunction(Exception):
    """
    a custom pt swap decision function can be passed to swaphandler, when swaphandler tries
    to use this function in method ptswap() it will raise this exception if it fails
    """
    pass

def default_pt_decision_function(args_1,args_2):
    """
    return decision to accept or reject swap
    accept swap with probability min(1,e^{-E_1 beta_2 - E_2 beta_1}/e^{-E_1 beta_1 - E_2 beta_2})
    """
    # unpack args
    E_1,beta_1,beta_2 = args_1
    E_2,beta_2_,beta_1_ = args_2
    # use the redundant temperature information as a consistency check
    assert beta_1==beta_1_
    assert beta_2==beta_2_
    # compute swap probability
    prob_switch = min( 1, np.exp( (-E_1*beta_2 - E_2*beta_1 + E_1*beta_1 + E_2*beta_2) ) )
    # return decision
    return ( np.random.random()<prob_switch )

class swaphandler(object):
    """
    PARALLEL TEMPERING - MESSAGE PASSING INTERFACE class

    Arguments:
    ==========
    mpi_comm_ : MPI communication environment
    mpi_rank_ : process rank
    number_swaps : number of swap rounds that will be run

    Outline of program:
    ===================
    Each process is a single copy of the system. They each have a process id (their rank) and 
    a temperature id. When the simulation starts these are the same. At a pt exchange step, 
    sets of two processes next to each other in temperature exchange mpi messages allowing them
    to work out whether they will swap temperature or not. 

    When we make a SWAP we do not move the copies of the systems, instead we exchange the 
    temperature indexes. Pointers to other processes keep track of the processes that neighbour 
    this one in temperatures. At each swap the pointers are exchanged. If for example we had 
    initially had, (process id):(temp id)
    
    1:1 <-> 2:2 <-> 3:3 <-> 4:4 <-> 5:5   &  2,3 exchange temperature, this goes to:
    
    1:1 <-> 3:2 <-> 2:3 <-> 4:4 <-> 5:5   but we want to keep the processes (copies of sys)
                                          in place so what we actually do is:
    1:1 <---------> 3:2 <-
                          |               (to make sure that 1 also points to 3, rather than
         -----------------                continuing to point to 2, we also need to SYNC)
        |
         -> 2:3 <---------> 4:4 <-> 5:5   the processes stay in place and all we do is
                                          adjust pointers

    Each round of swaps we select a subset of pairs (1,2),(3,4),etc. or (2,3),(4,5),etc. These
    are chosen by a list of random numbers 0 or 1 generated at the beginning of the program. The
    process with rank 0 generates this list and broadcasts it to other processes. During the swap
    step only the paired processes communicate.

    When we switch between the subsets an additional SYNC step is needed. If in the next step
    in the example above the subset changes from (2,3),(4,5),etc. -> (1,2),(3,4),etc. then the
    up-pointer from 1 has to be synced with the pair (2,3). 

    Stored properties:
    ==================
    self.decision_func : function to call to decide swap acceptance
    -----
    self.mpi_comm_world : MPI communication environment
    self.mpi_process_rank : process rank
    self.beta_index : index in the temperature set of this process's current temperature
    -----
    (SWAP STEP VARIABLES)
    self.pt_subsets : list of 'swap paritys' indicating how processes are paired for swaps at each round
    self.mpi_process_up_pointer : process rank of the system at the temperature above
    self.mpi_process_down_pointer : process rank of the system with the temperature below
    -----
    (SYNC STEP VARIABLES)
    self.prev_pt_subset : used to detect the need for a sync step
    self.mpi_sync_step_pointer : pointer to the process this will communicate with in the next sync step
    self.mpi_sync_pointer_direction : information about whether this pointer points up or down

    """
    def __init__( self, mpi_comm_,mpi_rank_, number_swaps=10000, disable_swaps=False, decision_func=None, verbose=False ):
        """
        """
        # store the self.mpi_comm_world and the process rank
        self.mpi_comm_world = mpi_comm_
        self.mpi_process_rank = mpi_rank_

        # set the pt decision function
        self.decision_func = default_pt_decision_function
        if not decision_func is None:
            self.decision_func = decision_func

        # make decision func a trivial no if disable_swaps
        if disable_swaps:
            def always_no(*args,**kwargs):
                return False
            self.decision_func = always_no

        # (if self.verbose) blank log file if it exists
        self.verbose = verbose
        if self.verbose:
            with open('log-file_rank'+str(self.mpi_process_rank)+'.txt', 'w+') as log_file:
                log_file.write('')

        # initialise all vars
        self.reset(number_swaps)

    def _log_vars( self ):
        """
        dump the value of all internal variables to the log file, for debugging
        """
        with open('log-file_rank'+str(self.mpi_process_rank)+'.txt', 'a') as log_file:
            # print all vars
            log_file.write('-'*10+'\n')
            log_file.write('beta_index : '+str(self.beta_index)+'\n')
            log_file.write('mpi_process_up_pointer : '+str(self.mpi_process_up_pointer)+'\n')
            log_file.write('mpi_process_down_pointer : '+str(self.mpi_process_down_pointer)+'\n')
            log_file.write('prev_pt_subset : '+str(self.prev_pt_subset)+'\n')
            log_file.write('mpi_sync_step_pointer : '+str(self.mpi_sync_step_pointer)+'\n')
            log_file.write('mpi_sync_pointer_direction : '+str(self.mpi_sync_pointer_direction)+'\n')

    def _init_pt_subsets( self,number_swaps ):
        """
        BROADCAST PT-SUBSETS LIST
        process 0 pre-generates the list of which random subset the pt exchanges occur
        within during each round, these are then broadcasted to all the other processes.
        """
        if ( self.mpi_process_rank == 0 ):
            self.pt_subsets = np.random.randint(2,size=number_swaps)
            # turn into bool array
            #self.pt_subsets = np.logical_and(self.pt_subsets,self.pt_subsets) 
        else:
            self.pt_subsets = np.empty(number_swaps,dtype=int)
            #self.pt_subsets = np.empty(number_swaps,dtype=np.bool)
        self.mpi_comm_world.Bcast([self.pt_subsets,MPI.INT], root=0)
        #self.mpi_comm_world.Bcast([self.pt_subsets,MPI.BOOL], root=0)
        # convert pt_subsets to regular array
        self.pt_subsets = self.pt_subsets.tolist()

    def reset( self,number_swaps ):
        """
        set all variables to initial state
        """
        # store the current temperature of this process as its position in the list of temperatures
        self.beta_index = self.mpi_process_rank

        # initialise pt-subsets shared resource
        self._init_pt_subsets(number_swaps)

        """
        INITIALISE POINTERS TO NEIGHBOURING PROCESSES
        each process has two pointers pointing to the processes running the
        temperature directly above and below this process. This is used to
        identify proper pairs in the PT rounds
        We use -1 to indicate that the process is at an end of the chain
        """
        self.mpi_process_up_pointer = int(self.mpi_process_rank)+1
        if ( self.mpi_process_up_pointer == int(self.mpi_comm_world.Get_size()) ):
            self.mpi_process_up_pointer = -1
        self.mpi_process_down_pointer = int(self.mpi_process_rank)-1

        if self.verbose:
            with open('log-file_rank'+str(self.mpi_process_rank)+'.txt', 'a') as log_file:
                #log_file.write('at temp '+str(self.beta_index)+' initial self.mpi_process_up_pointer '+str(self.mpi_process_up_pointer)+'\n')
                #log_file.write('at temp '+str(self.beta_index)+' initial self.mpi_process_down_pointer '+str(self.mpi_process_down_pointer)+'\n')
                log_file.write('at temp '+str(self.beta_index)+'\n')
                log_file.write('points '+str(self.mpi_process_down_pointer)+'<- . ->'+str(self.mpi_process_up_pointer)+'\n')

        """
        INITIALISE SYNC STEP POINTERS
        """
        self.prev_pt_subset = -1
        if (self.beta_index%2 == self.pt_subsets[-1]):
            # points down to the top of the pair below
            self.mpi_sync_step_pointer = self.mpi_process_down_pointer
            self.mpi_sync_pointer_direction = 0
        else:
            # points up to the bottom of the pair above
            self.mpi_sync_step_pointer = self.mpi_process_up_pointer
            self.mpi_sync_pointer_direction = 1

    def dump( self,file_handle ):
        """
        dump internal state as dict to json 
        exclude mpi-comm-environment and pt-subsets var as these are specific to instances
        """
        _state = {}
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("__") and not (attr=='mpi_comm_world') and not (attr=='pt_subsets'):
                _state[attr] = getattr(self,attr)
        json.dump(_state,file_handle)

    def load( self,file_handle ):
        """
        update the internal state to values set by json read
        exclude mpi-comm-environment and pt-subsets shared resource as these only really make sense at runtime
        raise an exception if the rank of this running process does not match the MPI rank in the loaded data 
        """
        _state = json.load(file_handle)
        if not (_state['mpi_process_rank']==self.mpi_process_rank):
            print('PtMPI load must match the current MPI process ranks to the loaded data')
            raise KeyError

        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("__") and not (attr=='mpi_comm_world') and not (attr=='pt_subsets'):
                try:
                    setattr(self,attr,_state[attr])
                except KeyError:
                    print('Aborting PtMPI load, data was missing field: '+attr)
                    self.reset()
                    return

    def get_current_temp_index( self ):
        """ 
        return the current temperature index of this rank
        """
        return self.beta_index

    def get_alternative_temp_index( self ):
        """ 
        return the possible alternative temperature index of this rank, that it could
        swap to in a pt swap step
        """
        try:
            _next_pt_subset = self.pt_subsets[-1]
        except IndexError:
            raise NoMoreSwaps

        if (self.beta_index%2 == _next_pt_subset) and (not (self.mpi_process_up_pointer == -1)):
            return self.beta_index + 1
        elif (not ( self.beta_index%2 == _next_pt_subset )) and (not (self.mpi_process_down_pointer == -1)):
            return self.beta_index - 1
        else:
            return 0

    def pt_step( self,*swap_decision_args_ ):
        """
        carry out parallel tempering swap round
        """
        #if self.verbose:
            #self._log_vars()

        # sync prcesses if needed
        self.pt_sync()

        # carry out swaps and return success flag
        return self.pt_swap(swap_decision_args_)

    def pt_sync( self ):
        """
        sync information between paired processes if necessary (see main docstring at top of file)
        """
        # outer try-finally loop ensures log_file is closed, but means it only it opens if
        # self.verbose is True
        try:
            if self.verbose:
                log_file = open('log-file_rank'+str(self.mpi_process_rank)+'.txt', 'a')
                log_file.write('-'*10+'\n')
                log_file.write('at sync'+'\n')

            try:
                _next_pt_subset = self.pt_subsets[-1]
            except IndexError:
                raise NoMoreSwaps

            if (not ( self.prev_pt_subset == _next_pt_subset )) and (not ( self.prev_pt_subset == -1 )):
                if self.verbose:
                    log_file.write('preparing to sync...'+'\n')

                if self.verbose:
                    if self.mpi_sync_pointer_direction==0:
                        log_file.write(str(self.mpi_process_rank)+' seeking to sync with '+str(self.mpi_sync_step_pointer)+'<-'+'\n')
                    else:
                        log_file.write(str(self.mpi_process_rank)+' seeking to sync with '+'->'+str(self.mpi_sync_step_pointer)+'\n')

                # handle the processes that began the last pt subset at the bottom of the pair, i.e. that are syncing
                # with the pair below
                if self.mpi_sync_pointer_direction == 0:
            
                    # identify if during the last pt subset this process instead ended up on the
                    # top of the pair
                    _process_has_swapped_during_last_subset = not ( self.beta_index%2 == self.prev_pt_subset )
            
                    # if the sync step pointer is -1 this indicates the process is at the lowest temperature and has no
                    # pair to sync with below, so nothing needs to be done
                    if not ( self.mpi_sync_step_pointer == -1 ):
                
                        # wait for a signal from the process at the end of the sync pointer
                        if self.verbose:
                            log_file.write('at temp '+str(self.beta_index)+' waiting for info from '+str(self.mpi_sync_step_pointer)+'\n')
                        incoming_data = np.empty(1, dtype=int)
                        self.mpi_comm_world.Recv( [incoming_data,MPI.INT], source=self.mpi_sync_step_pointer, tag=14 )
                
                        # *** -1 flags that the that process that sent the signal is still at the top of the pair below ***
                        # any other value tells us the process rank of the process that is now instead at the top
                        _new_top_process_subset_below = incoming_data[0]
                
                        # send a message back encoding whether or not this process is still at the bottom
                        # of the pair, if it is not send the process rank of the process that now is
                        if self.verbose:
                            log_file.write('at temp '+str(self.beta_index)+' sending info to '+str(self.mpi_sync_step_pointer)+'\n')
                        if _process_has_swapped_during_last_subset:
                            outgoing_data = np.array([ self.mpi_process_down_pointer ])
                        else:
                            outgoing_data = np.array([-1])
                        self.mpi_comm_world.Send( [outgoing_data,MPI.INT], dest=self.mpi_sync_step_pointer, tag=16 )
                
                    else:
                        _new_top_process_subset_below = -1
            
                    # if this process ended up on the top of the pair we need to sync within the pair
                    if _process_has_swapped_during_last_subset:
            
                        # wait for a message from the process paired process telling this process what
                        # its up-pointer should be pointing to
                        if self.verbose:
                            log_file.write('at temp '+str(self.beta_index)+' waiting for info from paired process '+str(self.mpi_process_down_pointer)+'\n')
                        incoming_data = np.empty(1, dtype=int)
                        self.mpi_comm_world.Recv( [incoming_data,MPI.INT], source=self.mpi_process_down_pointer, tag=18 )
                        if incoming_data[0]!=-1:
                            self.mpi_process_up_pointer = incoming_data[0]
            
                        # send a message to the processes paired process in the pair containing what its
                        # down-pointer should be pointing to
                        if self.verbose:
                            log_file.write('at temp '+str(self.beta_index)+' sending info to paired process '+str(self.mpi_process_down_pointer)+'\n')
                        outgoing_data = np.array([_new_top_process_subset_below])
                        self.mpi_comm_world.Send( [outgoing_data,MPI.INT], dest=self.mpi_process_down_pointer, tag=20 )
            
                    # else this process just has to update its down pointer
                    else:
                        if _new_top_process_subset_below!=-1:
                            self.mpi_process_down_pointer = _new_top_process_subset_below
            
                # handle the processes that began the last pt subset at the top of the pair, i.e. that are syncing
                # with the pair above
                else:
            
                    # identify if during the last pt subset this process instead ended up on the
                    # bottom of the pair
                    _process_has_swapped_during_last_subset = ( self.beta_index%2 == self.prev_pt_subset )
            
                    # if the sync step pointer is -1 this indicates the process is at the highest temperature and has no
                    # pair to sync with above, so nothing needs to be done
                    if not ( self.mpi_sync_step_pointer == -1 ):
            
                        # send a message to the process at the other end of the sync pointer telling it
                        # whether this process is still on the top of the pair or not
                        if self.verbose:
                            log_file.write('at temp '+str(self.beta_index)+' sending info to '+str(self.mpi_sync_step_pointer)+'\n')
                        if _process_has_swapped_during_last_subset:
                            outgoing_data = np.array([ self.mpi_process_up_pointer ])
                        else:
                            outgoing_data = np.array([-1])
                        self.mpi_comm_world.Send( [outgoing_data,MPI.INT], dest=self.mpi_sync_step_pointer, tag=14 )
                
                        # wait for a message back telling us whether the process we sent the message to
                        # is still at the bottom of the pair or not
                        if self.verbose:
                            log_file.write('at temp '+str(self.beta_index)+' waiting for info from '+str(self.mpi_sync_step_pointer)+'\n')
                        incoming_data = np.empty(1, dtype=int)
                        self.mpi_comm_world.Recv( [incoming_data,MPI.INT], source=self.mpi_sync_step_pointer, tag=16 )
                
                        # *** -1 flags that the that process that sent the signal is still at the bottom of the pair above ***
                        # any other value tells us the process rank of the process that is now instead at the bottom
                        _new_bottom_process_subset_above = incoming_data[0]
                
                    else:
                        _new_bottom_process_subset_above = -1
            
                    # if this process ended up on the bottom of the pair we need to sync within the pair
                    if _process_has_swapped_during_last_subset:
            
                        # send a message to the processes paired process in the pair containing what its
                        # up-pointer should be pointing to
                        if self.verbose:
                            log_file.write('at temp '+str(self.beta_index)+' sending info to paired process '+str(self.mpi_process_down_pointer)+'\n')
                        outgoing_data = np.array([_new_bottom_process_subset_above])
                        self.mpi_comm_world.Send( [outgoing_data,MPI.INT], dest=self.mpi_process_up_pointer, tag=18 )
            
                        # wait for a message from the process paired process telling this process what
                        # its down-pointer should be pointing to
                        if self.verbose:
                            log_file.write('at temp '+str(self.beta_index)+' waiting for info from paired process '+str(self.mpi_process_down_pointer)+'\n')
                        incoming_data = np.empty(1, dtype=int)
                        self.mpi_comm_world.Recv( [incoming_data,MPI.INT], source=self.mpi_process_up_pointer, tag=20 )
                        if not ( incoming_data[0] == -1 ):
                            self.mpi_process_down_pointer = incoming_data[0]
                    
                    # else this process just has to update its up-pointer
                    else:
                        if _new_bottom_process_subset_above!=-1:
                            self.mpi_process_up_pointer = _new_bottom_process_subset_above
            
                # set sync pointers for next sync round
                if ( self.beta_index%2 == _next_pt_subset ):
                    if self.verbose:
                        log_file.write('points '+str(self.mpi_process_down_pointer)+'<- [->'+str(self.mpi_process_up_pointer)+']'+'\n')
                    
                    # points down to the top of the pair below
                    self.mpi_sync_step_pointer = self.mpi_process_down_pointer
                    self.mpi_sync_pointer_direction = 0
                else:
                    if self.verbose:
                        log_file.write('points ['+str(self.mpi_process_down_pointer)+'<-] ->'+str(self.mpi_process_up_pointer)+'\n')

                    # points up to the bottom of the pair above
                    self.mpi_sync_step_pointer = self.mpi_process_up_pointer
                    self.mpi_sync_pointer_direction = 1
            else:
                if self.verbose:
                    log_file.write('nothing to do...'+'\n')
        
        except:
            raise
        finally:
            if self.verbose:
                log_file.close()

    def pt_swap( self,swap_decision_args_ ):
        """
        decide whether two processes should exchange temperature
        """
        # outer try-finally loop ensures log_file is closed, but means it only it opens if
        # self.verbose is True
        try:
            if self.verbose:
                log_file = open('log-file_rank'+str(self.mpi_process_rank)+'.txt', 'a')
                log_file.write('-'*10+'\n')
                log_file.write('at swap'+'\n')
                log_file.write('remaining available swaps '+str(len(self.pt_subsets))+'\n')

            _curr_pt_subset = self.pt_subsets.pop()
            self.prev_pt_subset = _curr_pt_subset

            if (self.beta_index%2==_curr_pt_subset) and (not (self.mpi_process_up_pointer == -1)):
                # controller chain at T recieves data from T+1 and makes a decision.

                # use the length of swap_decision_args_ to determine length of incoming info
                incoming_data = np.empty(2+len(swap_decision_args_), dtype=float)
                if self.verbose:
                    log_file.write('at temp '+str(self.beta_index)+' waiting for info from '+str(self.mpi_process_up_pointer)+' at temp '+str(self.beta_index+1)+'\n')
                try:
                    self.mpi_comm_world.Recv( [incoming_data,MPI.FLOAT], source=self.mpi_process_up_pointer, tag=10 )
                except OverflowError:
                    print('at temp '+str(self.beta_index)+ ' incoming_data '+ incoming_data)
                    print('at temp '+str(self.beta_index)+ ' self.mpi_process_up_pointer '+ self.mpi_process_up_pointer)
                    raise
                if self.verbose:
                    log_file.write('at temp '+str(self.beta_index)+' recieved info from '+str(self.mpi_process_up_pointer)+'\n')

                # unpack incoming data
                _TplusOne_up_pointer = int(incoming_data[0])
                _TplusOne_down_pointer = int(incoming_data[1])
                _other_swap_decision_args = tuple(incoming_data[2:])

                # decide whether to make pt switch
                try:
                    _pt_switch_decision = self.decision_func( swap_decision_args_,_other_swap_decision_args )
                except:
                    raise BadDecisionFunction('Could not use the decision function.')

                # send decision to paired process
                sending_data = np.array([ _pt_switch_decision, self.mpi_process_up_pointer, self.mpi_process_down_pointer ])
                if self.verbose:
                    log_file.write('at temp '+str(self.beta_index)+' sending decision to '+str(self.mpi_process_up_pointer)+' at temp '+str(self.beta_index+1)+'\n')
                self.mpi_comm_world.Send( [sending_data,MPI.INT], dest=self.mpi_process_up_pointer, tag=12 )

                # if swap was accepted handle change
                if _pt_switch_decision:
                    if self.verbose:
                        log_file.write('SWAP!'+'\n')
                        log_file.write('changing temp from '+str(self.beta_index)+' to '+str(self.beta_index+1)+'\n')
                    
                    # increase temperature index from T->T+1
                    self.beta_index += 1
                    
                    # update pointers
                    self.mpi_process_down_pointer = self.mpi_process_up_pointer
                    self.mpi_process_up_pointer = _TplusOne_up_pointer
                    if self.verbose:
                        log_file.write('points ['+str(self.mpi_process_down_pointer)+'<-] ->'+str(self.mpi_process_up_pointer)+'\n')
                else:
                    if self.verbose:
                        log_file.write('NO SWAP.'+'\n')

                return _pt_switch_decision

            elif (not ( self.beta_index%2 == _curr_pt_subset )) and (not ( self.mpi_process_down_pointer == -1 )):
                # non-controller chain at T+1 sends data to T and waits for decision
                
                # pack data for sending to T-1, this is [self.mpi_process_up_pointer, self.mpi_process_down_pointer, swap_decision_args]
                # swap_decision_args will be in the form of a tuple so convert it to an array
                sending_data = np.array([np.float64(self.mpi_process_up_pointer),np.float64(self.mpi_process_down_pointer)]+list(swap_decision_args_))
                if self.verbose:
                    log_file.write('at temp '+str(self.beta_index)+' sending info to '+str(self.mpi_process_down_pointer)+' at temp '+str(self.beta_index-1)+'\n')
                try:
                    self.mpi_comm_world.Send( [sending_data,MPI.FLOAT], dest=self.mpi_process_down_pointer, tag=10 )
                except OverflowError:
                    print('at temp '+str(self.beta_index)+ ' sending_data '+ sending_data)
                    print('at temp '+str(self.beta_index)+ ' self.mpi_process_down_pointer '+ self.mpi_process_down_pointer)
                    raise

                # wait for decision data
                decision_data = np.empty(3, dtype=int)
                if self.verbose:
                    log_file.write('at temp '+str(self.beta_index)+' waiting for decision from '+str(self.mpi_process_down_pointer)+' at temp '+str(self.beta_index-1)+'\n')
                self.mpi_comm_world.Recv( [decision_data,MPI.INT], source=self.mpi_process_down_pointer, tag=12 )
                if self.verbose:
                    log_file.write('at temp '+str(self.beta_index)+' recieved decision from '+str(self.mpi_process_down_pointer)+'\n')
                    
                # unpack decision data
                _pt_switch_decision = decision_data[0]
                _TminusOne_up_pointer = decision_data[1]
                _TminusOne_down_pointer = decision_data[2]
                    
                # if swap was accepted handle change
                if _pt_switch_decision:
                    if self.verbose:
                        log_file.write('SWAP!'+'\n')
                        log_file.write('changing temp from self.beta_index='+str(self.beta_index)+' to self.beta_index='+str(self.beta_index-1)+'\n')
                    
                    # decrease temperature index from T+1->T
                    self.beta_index -= 1
                    
                    # update pointers
                    self.mpi_process_up_pointer = self.mpi_process_down_pointer
                    self.mpi_process_down_pointer = _TminusOne_down_pointer
                    if self.verbose:
                        log_file.write('points '+str(self.mpi_process_down_pointer)+'<- [->'+str(self.mpi_process_up_pointer)+']'+'\n')
                else:
                    if self.verbose:
                        log_file.write('NO SWAP.'+'\n')
            else:
                if self.verbose:
                    log_file.write('nothing to do...'+'\n')

            return None

        except:
            raise
        finally:
            if self.verbose:
                log_file.close()
