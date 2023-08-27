
def typing_prior(
    bitvec,
    bsm,
    type_i,
    theta=1.,
    alpha=1.,
    ):

    from . import arrays
    import numpy as np

    _bitvec = arrays.bitvec(bitvec, bsm.maxbits)
    on = len(_bitvec.on())/float(bsm.maxbits)

    prior = -np.log(type_i+1.) -alpha*on + np.log(theta)

    return prior


def draw_bitvector_from_candidate_list(
    pvec, 
    bitvec_type_list, 
    bitsmarts_manager,
    targetcomputer,
    N_iter = 10,
    max_on=0.1,
    theta=1.,
    alpha=1.,
    verbose=False):

    """
    Sample bitvector types using bitsmarts_manager. Returns final bitvector list 
    after N_iter iterations.
    """

    import numpy as np
    import copy
    from . import arrays
    from .bitvector_typing import bitvec_hierarchy_to_allocations
    from .optimizers import batch_likelihood_typing

    N_types = pvec.force_group_count
    if N_types == 1:
        return
    
    _, bitvec_list_alloc_dict = bitsmarts_manager.prepare_bitvectors()

    type_schedule = np.arange(N_types, dtype=int)
    
    ### p = 1/t * exp(-alpha*on) * theta
    prior_func = lambda x: -np.log(x[0]+1.) -alpha*x[1] + np.log(theta)

    logPrior_all_list = np.zeros(N_types, dtype=float)
    for type_i in type_schedule:
        b_old = bitvec_type_list[type_i]
        _b_old = arrays.bitvec(b_old, bitsmarts_manager.maxbits)
        on = len(_b_old.on())/float(bitsmarts_manager.maxbits)
        logPrior_all_list[type_i] = prior_func((type_i,on))
        
    if verbose:
        print(
            "Generating candidates ..."
        )
    alloc_dict, smarts_dict, on_dict, subset_dict, bitvec_dict = bitsmarts_manager.and_rows(
        max_iter = 3,
        generate_smarts = False,
        max_neighbor = 3,
        max_on = max_on,
        duplicate_removal = False,
        verbose = verbose,
        )
    for t in alloc_dict:
        alloc_dict[t] = set(alloc_dict[t])
    
    for type_i in range(N_types):
        b = bitvec_type_list[type_i]
        _b_old = arrays.bitvec(
            b, bitsmarts_manager.maxbits
        )
        bitvec_dict[-type_i] = int(bitvec_type_list[type_i])
        on_dict[-type_i]     = len(_b_old.on())/float(bitsmarts_manager.maxbits)
        alloc_dict[-type_i]  = set()
        for a in bitvec_list_alloc_dict:
            if b == (a & b):
                alloc_dict[-type_i].update(
                    set(
                        bitvec_list_alloc_dict[a]
                        )
                    )

    for _ in range(N_iter):
        np.random.shuffle(type_schedule)
        for type_i in type_schedule:
            
            if verbose:
                print(
                    f"Sampling type {type_i}"
                )

            if verbose:
                print(
                    f"Scanning {len(bitvec_dict)} candidates ..."
                )
            b_old = int(bitvec_type_list[type_i])

            allocations_list = list()
            proposed_b_list  = []
            proposed_on_list = []
            allocations_map  = []

            allocations = [-1 for _ in pvec.allocations]
            bitvec_hierarchy_to_allocations(
                bitvec_list_alloc_dict,
                bitvec_type_list,
                allocations
                )
            selection_i = set()
            for type_j in range(type_i+1):
                for i, a in enumerate(allocations):
                    if a == type_j:
                        selection_i.add(i)

            for idx in bitvec_dict:
                check = alloc_dict[idx].issubset(selection_i)
                isold = bitvec_dict[idx] == b_old
                if isold or check:                
                    bitvec_type_list[type_i] = int(bitvec_dict[idx])
                    allocations = [-1 for _ in pvec.allocations]
                    bitvec_hierarchy_to_allocations(
                        bitvec_list_alloc_dict,
                        bitvec_type_list,
                        allocations
                        )
                    if allocations.count(-1) == 0:
                        if bitvec_dict[idx] not in proposed_b_list:
                            allocations = tuple(allocations)
                            allocations_list.append(allocations)
                            proposed_b_list.append(int(bitvec_dict[idx]))
                            proposed_on_list.append(on_dict[idx])
                            if allocations not in allocations_map:
                                allocations_map.append(allocations)
            
            if verbose:
                print(
                    f"Computing LogLikelihood for {len(allocations_map)} candidates ..."
                )
            proposed_on_list = np.array(proposed_on_list)
            _logL_list = batch_likelihood_typing(
                pvec,
                targetcomputer,
                allocations_map,
                rebuild_from_systems=False
                )
            if verbose:
                print(
                    f"Evaluating LogPosterior for {len(allocations_list)} candidates ..." 
                )

            logL_list = list()
            logPrior_list = list()
            for allocation, on in zip(allocations_list, proposed_on_list):
                idx = allocations_map.index(allocation)
                logL_list.append(_logL_list[idx])
                logPrior_list.append(
                    prior_func((type_i,on))
                )
                
            logL_list     = np.array(logL_list)
            logPrior_list = np.array(logPrior_list)
            logP_list     = np.copy(logL_list)
            logP_list    += logPrior_list
            for type_j in type_schedule:
                if type_j != type_i:
                    logP_list += logPrior_all_list[type_j]

            logP_sum = logP_list[0]
            for p in logP_list[1:]:
                logP_sum = np.logaddexp(p, logP_sum)
            logP_list_norm = logP_list - logP_sum
            P_list = np.exp(logP_list_norm)

            b_new = np.random.choice(
                proposed_b_list,
                size=None, 
                p=P_list
                )
            b_new = int(b_new)
            bitvec_type_list[type_i] = b_new
            idx_new = proposed_b_list.index(b_new)
            logPrior_all_list[type_i] = logPrior_list[idx_new]
            if verbose:
                sma_old = bitsmarts_manager.bitvector_to_smarts(b_old)
                sma_new = bitsmarts_manager.bitvector_to_smarts(b_new)
                idx_new = proposed_b_list.index(b_new)
                idx_old = proposed_b_list.index(b_old)

                print(
                    f"Type sweep {sma_old} --> {sma_new}\n",
                    f"logLikelihood: {logL_list[idx_old]:6.4f} --> {logL_list[idx_new]:6.4f}\n",
                    f"logPrior: {logPrior_list[idx_old]:6.4f} --> {logPrior_list[idx_new]:6.4f}\n",
                    f"logPosterior: {logP_list[idx_old]:6.4f} --> {logP_list[idx_new]:6.4f}",
                )
            
            if verbose:
                print(
                    f"New Typing hierarchy:"
                )
                allocations = allocations_list[idx_new]
                for idx, b in enumerate(bitvec_type_list):
                    sma = bitsmarts_manager.bitvector_to_smarts(b)
                    print(
                        f"Type {idx}({allocations.count(idx)})", sma
                    )
                print()

    return bitvec_type_list


def draw_bitvector_from_bits(
    pvec, 
    bitvec_type_list, 
    bitsmarts_manager,
    targetcomputer,
    N_iter = 10,
    bits_per_iter = 2,
    max_on=0.1,
    theta=2.,
    verbose=False):

    """
    Samples bitvector types bit-by-bit. Often generates complicated (and sometimes 
    chemically not sane) bitvectors. Returns final bitvector list after N_iter iterations. 
    """

    import numpy as np
    from . import arrays
    import copy
    import itertools
    from .bitvector_typing import bitvec_hierarchy_to_allocations
    from .optimizers import batch_likelihood_typing

    N_types = pvec.force_group_count
    if N_types == 1:
        return
    
    _, bitvec_list_alloc_dict = bitsmarts_manager.prepare_bitvectors()

    type_schedule = np.arange(N_types, dtype=int)
    
    logPrior_all_list = np.zeros(N_types, dtype=float)
    for type_i in type_schedule:
        b_old = bitvec_type_list[type_i]
        _b_old = arrays.bitvec(b_old, bitsmarts_manager.maxbits)
        on = len(_b_old.on())/float(bitsmarts_manager.maxbits)
        logPrior_all_list[type_i] = -np.log(type_i+1.) + np.log(1.-on) + np.log(theta)
        
    bit_mask = list(itertools.product([0,1], repeat=bits_per_iter))

    for _ in range(N_iter):
        np.random.shuffle(type_schedule)
        for type_i in type_schedule:
            if verbose:
                print(
                    f"Sampling type {type_i}"
                )

            bitvec_schedule = np.arange(bitsmarts_manager.maxbits, dtype=int)
            np.random.shuffle(bitvec_schedule)
            start = 0
            stop  = bits_per_iter
            while start < bitsmarts_manager.maxbits:
                b_old = bitvec_type_list[type_i]
                bitvec = arrays.bitvec(
                    b_old,
                    bitsmarts_manager.maxbits
                )

                _bitvec_schedule = bitvec_schedule[start:stop].tolist()
                start = stop
                stop += bits_per_iter
                if stop > bitsmarts_manager.maxbits:
                    stop = bitsmarts_manager.maxbits
                
                allocations_list = list()
                proposed_b_list  = list()
                proposed_on_list = list()

                for m in bit_mask:
                    for c, i in enumerate(_bitvec_schedule):
                        bitvec[i] = int(m[c])
                    b = bitvec.v
                    bitvec_type_list[type_i] = b
                    allocations = [-1 for _ in pvec.allocations]
                    bitvec_hierarchy_to_allocations(
                        bitvec_list_alloc_dict,
                        bitvec_type_list,
                        allocations
                        )
                    if allocations.count(-1) == 0:
                        if b not in proposed_b_list:
                            on = len(bitvec.on())/float(bitvec.maxbits)
                            allocations_list.append(allocations)
                            proposed_b_list.append(b)
                            proposed_on_list.append(on)

                bitvec_type_list[type_i] = b_old
                if len(proposed_on_list) == 0:                    
                    continue

                if verbose:
                    print(
                        f"Computing LogP for {len(proposed_on_list)} candidates ..."
                    )
                proposed_on_list = np.array(proposed_on_list)
                logL_list = batch_likelihood_typing(
                    pvec,
                    targetcomputer,
                    allocations_list,
                    rebuild_from_systems=False
                    )
                logL_list     = np.array(logL_list)
                logP_list     = np.copy(logL_list)
                logPrior_list = -np.log(type_i+1.) + np.log(1.-proposed_on_list) + np.log(theta)
                logP_list    += logPrior_list
                for type_j in type_schedule:
                    if type_j != type_i:
                        logP_list += logPrior_all_list[type_j]

                logP_sum = logP_list[0]
                for p in logP_list[1:]:
                    logP_sum = np.logaddexp(p, logP_sum)
                logP_list_norm = logP_list - logP_sum
                P_list = np.exp(logP_list_norm)

                b_new = np.random.choice(
                    proposed_b_list,
                    size=None, 
                    p=P_list
                    )
                b_new   = int(b_new)
                idx_new = proposed_b_list.index(b_new)
                logPrior_all_list[type_i] = logPrior_list[idx_new]
                if verbose:
                    idx_new = proposed_b_list.index(b_new)
                    idx_old = proposed_b_list.index(b_old)                    
                    try:
                        sma_old = bitsmarts_manager.bitvector_to_smarts(b_old)
                        sma_new = bitsmarts_manager.bitvector_to_smarts(b_new)
                        print(
                            f"Type sweep {sma_old} --> {sma_new}",
                        )
                    except:
                        print(
                            "Could not generate SMARTS string."
                        )
                    print(
                        f"logLikelihood: {logL_list[idx_old]:6.4f} --> {logL_list[idx_new]:6.4f}\n",
                        f"logPrior: {logPrior_list[idx_old]:6.4f} --> {logPrior_list[idx_new]:6.4f}\n",
                        f"logPosterior: {logP_list[idx_old]:6.4f} --> {logP_list[idx_new]:6.4f}",
                    )

                bitvec_type_list[type_i] = b_new
                if verbose:
                    #print(
                    #    f"New Typing hierarchy:"
                    #)
                    for idx, b in enumerate(bitvec_type_list):
                        try:
                            sma = bitsmarts_manager.bitvector_to_smarts(b)
                            print(
                                f"Type {idx}", sma
                            )
                        except:
                            print(
                                f"Type {idx}", "XXXX"
                            )                            
                    print()

    return bitvec_type_list