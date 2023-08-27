import ray

def get_primitives(rdmol_list, smarts_dict, atom_idx_list):

    from rdkit import Chem
    import numpy as np

    results_dict = dict()
    for key in smarts_dict:
        results_dict[key] = list()
        for rdmol in rdmol_list:
            matches = rdmol.GetSubstructMatches(
                smarts_dict[key],
                useQueryQueryMatches=False,
                uniquify=False,
            )
            if len(matches) > 0:
                matches = np.array(matches, dtype=int)[:,atom_idx_list]
            else:
                matches = np.array([[]], dtype=int)
            results_dict[key].append(
                matches
            )

    return results_dict


@ray.remote
def compute_bitvector_product(
    bit_list, 
    start_idx, 
    stop_idx, 
    max_on, 
    maxbits):

    import numpy as np
    from . import arrays
    
    bit_list_update = set()
    _bit_list = tuple(bit_list)
    for idx in range(start_idx, stop_idx):
        row = np.floor(-0.5 + np.sqrt(0.25 + 2 * idx))
        triangularNumber = row * (row + 1) / 2
        column = idx - triangularNumber

        idx1 = int(row)
        idx2 = int(column)

        b1 = _bit_list[idx1]
        b2 = _bit_list[idx2]
        b12 = b1 & b2
        _b12 = arrays.bitvec(b12, maxbits)
        on = len(_b12.on())/float(maxbits)
        
        if on < max_on:        
            if b12 not in _bit_list:
                bit_list_update.add(b12)

    return bit_list_update


@ray.remote
def prepare_encoding(
    b,
    bn,
    a,
    _max_neighbor,
    max_neighbor,
    primitive_mapping_neighbor_dict,
    primitive_mapping_dict,
    maxbits,
    max_on
    ):

    import itertools
    from . import arrays

    bitvec_list_alloc_dict = dict()
    bitvec_list = set()

    vec_length = len(primitive_mapping_dict)
    
    ### Select neighbor
    for n in itertools.permutations(range(_max_neighbor), max_neighbor):
        ### Select bit shift
        for i in itertools.permutations(range(_max_neighbor), max_neighbor):
            bnew_n = arrays.bitvec(0, maxbits=maxbits)
            bnew   = arrays.bitvec(0, maxbits=maxbits)
            for key in primitive_mapping_neighbor_dict:
                mapped_idx_list = primitive_mapping_neighbor_dict[key]
                for _n, _i in zip(n,i):
                    mapped_idx           = mapped_idx_list[_n]
                    mapped_idx_new       = mapped_idx_list[_i] + vec_length
                    bnew_n[mapped_idx_new] = bn[mapped_idx]
            for key in primitive_mapping_dict:
                mapped_idx     = primitive_mapping_dict[key]
                mapped_idx_new = primitive_mapping_dict[key]
                bnew_n[mapped_idx_new] = b[mapped_idx]
                bnew[mapped_idx_new]   = b[mapped_idx]
            on  = float(len(bnew_n.on()))
            on /= float(maxbits)
            if on < max_on:
                key = bnew.v
                key_n = bnew_n.v
                if key not in bitvec_list:
                    bitvec_list.add(key)
                if key_n not in bitvec_list:
                    bitvec_list.add(key_n)
                if key in bitvec_list_alloc_dict:
                    if a not in bitvec_list_alloc_dict[key]:
                        bitvec_list_alloc_dict[key].add(a)
                else:
                    bitvec_list_alloc_dict[key] = set([a])

                if key_n in bitvec_list_alloc_dict:
                    if a not in bitvec_list_alloc_dict[key_n]:
                        bitvec_list_alloc_dict[key_n].add(a)
                else:
                    bitvec_list_alloc_dict[key_n] = set([a])

    return bitvec_list, bitvec_list_alloc_dict


def and_bitvectors(
    bitvec_list, 
    max_iter,
    maxbits, 
    max_on,
    N_atoms,
    vec_length,
    primitive_mapping_dict,
    primitive_mapping_neighbor_dict,
    reverse_primitive_mapping_dict,
    reverse_primitive_mapping_neighbor_dict,
    verbose=False):

    _CHUNK_SIZE_MAX = 50000

    if len(bitvec_list) == 0:
        return set()
    elif len(bitvec_list) == 1:
        return set(bitvec_list)
    
    max_on = float(max_on)
    bitvec_list = set(bitvec_list)
    found_new = 1
    for _ in range(max_iter):
        ### If True, we have not
        ### found a new bitvec and therefore
        ### can stop.
        if found_new  == 0:
            break
        found_new = 0
        chunk_size_max = _CHUNK_SIZE_MAX
        N_keys   = len(bitvec_list)
        N_entries = N_keys * (N_keys - 1) / 2
        N_entries = int(N_entries)
        bitvec_list_id = ray.put(tuple(bitvec_list))
        if chunk_size_max > N_entries:
            chunk_size_max = N_entries
        start = 0
        stop  = chunk_size_max
        chunks_count = int(N_entries/chunk_size_max)
        worker_id_list = list()
        for _ in range(chunks_count):
            worker_id_list.append(
                compute_bitvector_product.remote(
                    bitvec_list_id,
                    start, 
                    stop,
                    max_on,
                    maxbits
                )
            )
            start = stop
            stop  = start + chunk_size_max
        res = N_entries%chunk_size_max
        if res > 0:
            start = N_entries-res
            stop = start + res
            worker_id_list.append(
                compute_bitvector_product.remote(
                    bitvec_list_id,
                    start, 
                    stop,
                    max_on,
                    maxbits
                )
            )

        while worker_id_list:
            [worker_id], worker_id_list = ray.wait(worker_id_list)
            bitvec_list_new = ray.get(worker_id)
            d_new = len(bitvec_list)
            bitvec_list.update(bitvec_list_new)
            d_new = len(bitvec_list) - d_new
            found_new += d_new
                    
        if verbose:
            print(
                f"Found {len(bitvec_list)} unique bit vectors."
            )
                    
    return bitvec_list


def sanitize_atom_smarts(sma):
    if "#1" in sma and "H0" in sma:
        sma = sma.replace("&H0", "")
        sma = sma.replace("H0", "")
    if "#1" in sma and "X1" in sma:
        sma = sma.replace("&X1", "")
        sma = sma.replace("X1", "")
    if "+0" in sma or "-0" in sma:
        sma = sma.replace("&+0", "")
        sma = sma.replace("&-0", "")
        sma = sma.replace("+0", "")
        sma = sma.replace("-0", "")
    if "r" in sma and "R" in sma:
        sma = sma.replace("R", "")
        sma = sma.replace("&R", "")
    if "!&" in sma:
        sma = sma.replace("!&", "")
    if "&:" in sma:
        sma = sma.replace("&:", "")
    if "&!:" in sma:
        sma = sma.replace("&!:", "")
        
    while sma.startswith("[&"):
        sma = "[" + sma[2:]
        
    return sma


class BitSmartsManager(object):
    
    import numpy as np
    
    aromatic_smarts = ["a", "A"]
    charge_smarts = ["-2", "-1", "+0", "+1", "+2"]
    #atom_smarts = ["!#1", "#1", "!#6", "#6", "#7", "#8", "#16"]
    atom_smarts = ["#1", "#6", "#7", "#8", "#16"]
    #hydrogen_smarts = ["H1", "H2", "H3", "!H0", "!H1", "!H2", "!H3"]
    hydrogen_smarts = ["H1", "H2", "H3"]
    #conn_smarts = ["X0", "X1", "X2", "X3", "X4", "!X0", "!X1", "!X2", "!X3", "!X4"]
    conn_smarts = ["X0", "X1", "X2", "X3", "X4"]
    #ring_smarts = ["!r", "!r3", "!r4", "r", "r3", "r4", "r5", "r6"]
    ring_smarts = ["r", "r3", "r4", "r5", "r6"]

    #bond_type = ["!-", "!=", "!#", "-", "=", "#"]
    bond_type = ["-", "=", "#"]
    #bond_ring = ["!@","@"]
    bond_ring = ["@"]
    bond_aromatic = [":"]

    atom_primitives_smarts  = atom_smarts[:]
    atom_primitives_smarts += charge_smarts[:]
    atom_primitives_smarts += aromatic_smarts[:]
    atom_primitives_smarts += hydrogen_smarts[:]
    atom_primitives_smarts += conn_smarts[:]
    atom_primitives_smarts += ring_smarts[:]
    
    bond_primitives_smarts  = bond_type[:]
    bond_primitives_smarts += bond_ring[:]
    bond_primitives_smarts += bond_aromatic[:]

    all_primitives_smarts  = atom_primitives_smarts[:]
    all_primitives_smarts += bond_primitives_smarts[:]

    def __init__(self, 
                 parameter_manager, 
                 parent_smarts = None, 
                 max_neighbor = 3
                ):

        ### By default
        use_neg = False
        
        from rdkit import Chem
        import numpy as np

        ### If parent is not specified,
        ### just take the most generic smarts.
        if parent_smarts == None:
            parent_smarts = "~".join(
                [f"[*:{i+1:d}]" for i in range(parameter_manager.N_atoms)]
                )

        ### Note:
        ### The code currently really only works
        ### with `max_layer = 1`. Anything else is
        ### not sensical. If more layers are requested,
        ### run the code recursively.
        max_layer = 1

        self._max_neighbor = max_neighbor
        
        assert max_neighbor <= 3
        assert max_neighbor > -1

        self.all_primitives_smarts += [parent_smarts][:]

        self.atom_primitives_smarts = np.array(self.atom_primitives_smarts)
        self.bond_primitives_smarts = np.array(self.bond_primitives_smarts)
        self.all_primitives_smarts  = np.array(self.all_primitives_smarts)

        self.N_atoms = parameter_manager.N_atoms

        self.N_allocs    = np.max(parameter_manager.force_ranks) + 1
        self.rdmol_list  = parameter_manager.rdmol_list
        self.atom_list   = np.array(parameter_manager.atom_list)
        self.rank_list   = np.array(parameter_manager.force_ranks)
        self.sysidx_list = np.array(parameter_manager.system_idx_list)

        self.parent_smarts = parent_smarts
        self.parent_rdmol  = Chem.MolFromSmarts(parent_smarts)
        bond_list = list()
        for bond in self.parent_rdmol.GetBonds():
            am1 = bond.GetBeginAtom().GetAtomMapNum()
            am2 = bond.GetEndAtom().GetAtomMapNum()
            if am1 > 0 and am2 > 0:
                am12 = sorted([am1, am2])
                am12 = tuple(am12)
                bond_list.append(am12)
        self.bond_list = np.array(bond_list, dtype=int)
        self.N_bonds   = len(bond_list)

        tagged_atom_dict         = dict()
        reverse_tagged_atom_dict = dict()
        atom_idx_list_parents = list()
        base_query_rdmol = Chem.RWMol()
        for atom in self.parent_rdmol.GetAtoms():
            tag = atom.GetAtomMapNum()           
            if tag > 0:
                tagged_atom_dict[tag] = atom.GetIdx()
                reverse_tagged_atom_dict[atom.GetIdx()] = tag
                atom_idx_list_parents.append(tag-1)
                
                atom_idx = base_query_rdmol.AddAtom(
                    Chem.AtomFromSmarts(f"[*:{tag}]")
                )
            else:
                atom_idx = base_query_rdmol.AddAtom(
                    Chem.AtomFromSmarts(f"[*]")
                )                
        for bond in bond_list:
            idx1 = tagged_atom_dict[bond[0]]
            idx2 = tagged_atom_dict[bond[1]]
            bond_idx = base_query_rdmol.AddBond(
                idx1,
                idx2,
            )
            base_query_rdmol.ReplaceBond(
                bond_idx-1,
                Chem.BondFromSmarts("~")
            )
        base_query_rdmol = base_query_rdmol.GetMol()
        self.base_query_rdmol = base_query_rdmol
        
        self.tagged_atom_dict      = tagged_atom_dict
        self.reverse_tagged_atom_dict = reverse_tagged_atom_dict
        self.atom_idx_list_parents = np.array(sorted(atom_idx_list_parents), dtype=int)
        
        assert len(self.tagged_atom_dict) == len(self.atom_idx_list_parents)

        self._bond_primitive_count = len(self.bond_primitives_smarts)
        self._atom_primitive_count = len(self.atom_primitives_smarts)
        self.smarts_dict = dict()
        
        self.smarts_dict[parent_smarts,0,0] = self.parent_rdmol
        
        ### PROCESS ATOMS
        ### =============
        for sma in self.atom_primitives_smarts:
            for layer in [0,1]:
                for tag in range(1,self.N_atoms+1):
                    atom_idx = self.tagged_atom_dict[tag]
                    query_rdmol = Chem.RWMol(base_query_rdmol)
                    if layer == 0:
                        sma_in = f"[{sma}:{tag}]"
                    else:
                        atom_idx_new = query_rdmol.AddAtom(
                            Chem.AtomFromSmarts(f"[*]")
                        )
                        bondidx = query_rdmol.AddBond(
                            atom_idx_new,
                            atom_idx,
                        )
                        query_rdmol.ReplaceBond(
                            bondidx-1,
                            Chem.BondFromSmarts("~")
                        )
                        sma_in = f"[{sma}]"
                        atom_idx = atom_idx_new
                    query_rdmol.ReplaceAtom(
                        atom_idx,
                        Chem.AtomFromSmarts(sma_in)
                    )
                    key = (sma, layer, tag)
                    self.smarts_dict[key] = query_rdmol.GetMol()
                    if use_neg:
                        if layer == 0:
                            sma_in = f"[!{sma}:{tag}]"
                        else:
                            sma_in = f"[!{sma}]"
                        query_rdmol.ReplaceAtom(
                            atom_idx,
                            Chem.AtomFromSmarts(sma_in)
                        )
                        key = (f"!{sma}", layer, tag)
                        self.smarts_dict[key] = query_rdmol.GetMol()

        ### PROCESS BONDS
        ### =============
        for sma in self.bond_primitives_smarts:
            for layer in [0,1]:
                if layer == 0:
                    for bond in bond_list:
                        bond = tuple(sorted(bond))
                        query_rdmol = Chem.RWMol(base_query_rdmol)
                        idx1 = self.tagged_atom_dict[bond[0]]
                        idx2 = self.tagged_atom_dict[bond[1]]
                        bond_idx = query_rdmol.GetBondBetweenAtoms(idx1, idx2).GetIdx()
                        query_rdmol.ReplaceBond(
                            bond_idx,
                            Chem.BondFromSmarts(f"{sma}")
                        )
                        key = (sma, layer, bond)
                        self.smarts_dict[key] = query_rdmol.GetMol()
                        if use_neg:
                            sma_in = f"!{sma}"
                            query_rdmol.ReplaceAtom(
                                atom_idx,
                                Chem.AtomFromSmarts(sma_in)
                            )
                            key = (f"!{sma}", layer, bond)
                            self.smarts_dict[key] = query_rdmol.GetMol()
                else:
                    for tag in range(1,self.N_atoms+1):
                        for sma in self.bond_primitives_smarts:
                            query_rdmol  = Chem.RWMol(base_query_rdmol)
                            atom_idx     = self.tagged_atom_dict[tag]
                            atom_idx_new = query_rdmol.AddAtom(
                                Chem.AtomFromSmarts(f"[*]")
                            )
                            bond_count = query_rdmol.AddBond(
                                atom_idx_new,
                                atom_idx,
                            )
                            query_rdmol.ReplaceBond(
                                bond_count-1,
                                Chem.BondFromSmarts(sma)
                            )
                            _tag = (tag, -1)
                            key = (sma, layer, _tag)
                            self.smarts_dict[key] = query_rdmol.GetMol()
                            if use_neg:
                                sma_in = f"!{sma}"
                                query_rdmol.ReplaceBond(
                                    bondidx-1,
                                    Chem.BondFromSmarts(sma_in)
                                )
                                key = (f"!{sma}", layer, _tag)
                                self.smarts_dict[key] = query_rdmol.GetMol()

        primitive_mapping_dict = dict()
        primitive_mapping_neighbor_dict = dict()
        reverse_primitive_mapping_dict = dict()
        reverse_primitive_mapping_neighbor_dict = dict()        
        vec_length = 0
        vec_length_neighbor = 0
        for key in self.smarts_dict:
            sma, layer, tag = key
            if sma == self.parent_smarts:
                continue
            if layer == 0:
                primitive_mapping_dict[key] = vec_length
                reverse_primitive_mapping_dict[vec_length] = key
                vec_length += 1
            else:
                primitive_mapping_neighbor_dict[key] = list()
                for n in range(self._max_neighbor):
                    primitive_mapping_neighbor_dict[key].append(
                        vec_length_neighbor
                    )
                    reverse_primitive_mapping_neighbor_dict[vec_length_neighbor] = key, n
                    vec_length_neighbor += 1
        self.primitive_mapping_dict = primitive_mapping_dict
        self.primitive_mapping_neighbor_dict = primitive_mapping_neighbor_dict
        self.reverse_primitive_mapping_dict = reverse_primitive_mapping_dict
        self.reverse_primitive_mapping_neighbor_dict = reverse_primitive_mapping_neighbor_dict
        
        self.vec_length_neighbor = vec_length_neighbor
        self.vec_length = vec_length
        self.maxbits = self.vec_length_neighbor + self.vec_length

        from . import arrays
        self.primitive_binary = [arrays.bitvec(0,maxbits=self.vec_length) for _ in range(self.N_allocs*2)]
        self.primitive_binary_neighbor = [arrays.bitvec(0,maxbits=self.vec_length_neighbor) for _ in range(self.N_allocs*2)]
        self.primitive_binary_parent = arrays.bitvec(0,maxbits=self.N_allocs)


    def generate(self, ring_safe=True):

        import numpy as np

        all_primitives_dict = get_primitives(
            self.rdmol_list,
            self.smarts_dict,
            np.append(
                self.atom_idx_list_parents,
                -1
            )
        )
        
        neighbor_indices = np.zeros(
            (
                self.N_allocs,
                self.N_atoms,
                self._max_neighbor
            ),
            dtype=np.int32
        )
        neighbor_indices[:] = -1
        for key in self.smarts_dict:
            sma, layer, tag = key
            for sys_idx in range(len(self.rdmol_list)):
                valids = np.where(self.sysidx_list == sys_idx)[0]
                if valids.size == 0:
                    continue
                _atom_list = np.copy(self.atom_list[valids])
                _rank_list = np.copy(self.rank_list[valids])

                _rank_list, unique_rank_idxs = np.unique(
                    _rank_list, 
                    return_index=True
                    )
                _atom_list = _atom_list[unique_rank_idxs]
                atom_idx_list = np.copy(all_primitives_dict[key][sys_idx])
                if atom_idx_list.size == 0:
                    continue
                    
                dtype = np.dtype(
                    'S{:d}'.format(
                        self.N_atoms * atom_idx_list.dtype.itemsize
                    )
                )

                #if layer > 0:
                #    invalids = np.equal(
                #        atom_idx_list[:,:-1],
                #        atom_idx_list[:,-1,np.newaxis]
                #    )
                #    invalids = np.any(invalids, axis=1)
                #    atom_idx_list = atom_idx_list[
                #        np.logical_not(invalids)
                #    ]                

                atom_idx_list_1 = np.frombuffer(
                    atom_idx_list[:,:-1].tobytes(), 
                    dtype=dtype
                )
                atom_idx_list_2 = np.frombuffer(
                    _atom_list.tobytes(), 
                    dtype=dtype
                )

                intersect, c0, c1 = np.intersect1d(
                    atom_idx_list_1, 
                    atom_idx_list_2, 
                    return_indices=True
                )

                if intersect.size == 0:
                    continue

                allocs, _alloc_rank = np.unique(
                    _rank_list[c1],
                    return_index=True
                )
                c0 = c0[_alloc_rank]
                c1 = c1[_alloc_rank]
                
                assert c0.size == allocs.size
                assert c0.size == c1.size

                if layer == 0:
                    for a in allocs:
                        a = int(a)
                        if sma == self.parent_smarts:
                            self.primitive_binary_parent[a] = 1
                        else:
                            mapped_idx = self.primitive_mapping_dict[key]
                            self.primitive_binary[a][mapped_idx] = 1
                elif self._max_neighbor > 0:
                    ### Bond
                    if isinstance(tag, (list,tuple)):
                        tag_idx = self.tagged_atom_dict[tag[0]]
                    ### Atom
                    else:
                        tag_idx = self.tagged_atom_dict[tag]
                    if tag_idx == -1:
                        continue
                    allocs_valids = allocs
                    c0_valids     = c0

                    valids = np.less(
                        neighbor_indices[allocs_valids,tag_idx],
                        0
                    )
                    _valids = np.equal(
                        atom_idx_list[c0_valids,-1,np.newaxis],
                        neighbor_indices[allocs_valids,tag_idx]
                    )
                    invalid_rows = np.any(_valids, axis=1)
                    valids[invalid_rows] = False

                    valid_rows = np.any(valids, axis=1)
                    col_idxs   = np.argmax(valids, axis=1)

                    if np.any(valids):
                        _atom_idx_list = atom_idx_list[c0_valids[valid_rows],-1]
                        if ring_safe:
                            ringsafe_neighbors = np.delete(
                                neighbor_indices[allocs_valids[valid_rows],:,:],
                                tag_idx,
                                axis=1
                            )
                            valids_not_ringsafe = np.equal(
                                _atom_idx_list[:,np.newaxis,np.newaxis],
                                ringsafe_neighbors
                            )
                            valids_ringsafe = np.any(
                                valids_not_ringsafe, 
                                axis=(1,2)
                            )
                            valids_ringsafe = np.logical_not(valids_ringsafe)
                            _allocs = allocs_valids[valid_rows][valids_ringsafe]
                            _cols   = col_idxs[valid_rows][valids_ringsafe]
                            _atom_idx_list = _atom_idx_list[valids_ringsafe]
                        else:
                            _allocs = allocs_valids[valid_rows]
                            _cols   = col_idxs[valid_rows]
                            _atom_idx_list = _atom_idx_list

                        neighbor_indices[_allocs,tag_idx,_cols] = _atom_idx_list

                    valids = np.equal(
                        atom_idx_list[c0_valids,-1,np.newaxis],
                        neighbor_indices[allocs_valids,tag_idx]
                    )
                    mapped_idx_list = self.primitive_mapping_neighbor_dict[key]
                    nrows = valids.shape[0]
                    for row_idx in range(nrows):
                        a = int(allocs_valids[row_idx])
                        for v in np.where(valids[row_idx])[0]:
                            mapped_idx = mapped_idx_list[v]
                            self.primitive_binary_neighbor[a][mapped_idx] = 1

        ### Now make sure that also the reverse
        ### of the smarts is mapped correctly.
        for key in self.smarts_dict:
            sma, layer, tag = key
            if sma == self.parent_smarts:
                continue
            if layer == 0:
                mapped_idx_old = self.primitive_mapping_dict[key]
                ### Bond
                if isinstance(tag, (list,tuple)):
                    tag1 = self.N_atoms - tag[0] + 1
                    tag2 = self.N_atoms - tag[1] + 1
                    tag  = sorted([tag1,tag2])
                    tag  = tuple(tag)
                ### Atom
                else:
                    tag = self.N_atoms - tag + 1
                key = sma, layer, tag
                mapped_idx_new = self.primitive_mapping_dict[key]
                for a_old in range(self.N_allocs):
                    a_new = a_old + self.N_allocs
                    self.primitive_binary[a_new][mapped_idx_new] = self.primitive_binary[a_old][mapped_idx_old]
            else:
                mapped_idx_n_old = self.primitive_mapping_neighbor_dict[key]
                ### Bond
                if isinstance(tag, (list,tuple)):
                    _tag = self.N_atoms - tag[0] + 1
                    tag_new = (_tag, -1)
                    tag = tag_new
                ### Atom
                else:
                    tag = self.N_atoms - tag + 1
                key = sma, layer, tag

                mapped_idx_n_new = self.primitive_mapping_neighbor_dict[key]
                for a_old in range(self.N_allocs):
                    a_new = a_old + self.N_allocs
                    for m_new, m_old in zip(mapped_idx_n_new, mapped_idx_n_old):
                        self.primitive_binary_neighbor[a_new][m_new] = self.primitive_binary_neighbor[a_old][m_old]


    def prepare_bitvectors(
        self,
        allocations = list(),
        verbose = False,
        max_neighbor = 3,
        max_on=1.):
        
        maxbits = self.maxbits

        max_on = float(max_on)

        if max_neighbor > self._max_neighbor:
            max_neighbor = self._max_neighbor
        assert max_neighbor > -1

        from . import arrays
        import itertools
        import ray

        if verbose:
            print(
                "Preparing bitvector encoding"
            )

        worker_id_list = list()
        primitive_mapping_neighbor_dict_id = ray.put(
            self.primitive_mapping_neighbor_dict
        )
        primitive_mapping_dict_id = ray.put(
            self.primitive_mapping_dict
        )

        if len(allocations) == 0:
            allocations = list(range(self.N_allocs))

        for a in allocations:
            v = self.primitive_binary_parent[a]
            if v:
                for _a in [a, a+self.N_allocs]:
                    b = self.primitive_binary[_a]
                    bn = self.primitive_binary_neighbor[_a]
                    worker_id_list.append(
                        prepare_encoding.remote(
                            b,
                            bn,
                            a, ### This must be `a`, not `_a`
                            self._max_neighbor,
                            max_neighbor,
                            primitive_mapping_neighbor_dict_id,
                            primitive_mapping_dict_id,
                            maxbits,
                            max_on
                        )
                    )

        bitvec_list = set()
        bitvec_list_alloc_dict = dict()
        while worker_id_list:
            worker_id, worker_id_list = ray.wait(worker_id_list)
            _bitvec_list, _bitvec_list_alloc_dict = ray.get(worker_id[0])
            bitvec_list.update(_bitvec_list)
            for key in _bitvec_list_alloc_dict:
                if key in bitvec_list_alloc_dict:
                    bitvec_list_alloc_dict[key].update(
                        _bitvec_list_alloc_dict[key]
                    )
                else:
                    bitvec_list_alloc_dict[key] = _bitvec_list_alloc_dict[key]
            
        return bitvec_list, bitvec_list_alloc_dict


    def _eliminate_duplicates(
        self, 
        bitvec_list, 
        bitvec_list_alloc_dict,
        maxbits):
        
        __doc__ = """
        Eliminate mirrored (in the SMARTS sense) entries
        """
        
        smarts_list     = set()
        bitvec_list_new = set()
        bitvec_list_alloc_dict_new = dict()

        from . import arrays
        import itertools

        b_old  = arrays.bitvec(0, maxbits)
        b_new  = arrays.bitvec(0, maxbits)
        for b in bitvec_list:
            b_old.v = b
            b_new.v = 0
            has_layer = False
            for i in range(maxbits):
                if not b_old[i]:
                    continue
                if i < self.vec_length:
                    sma, layer, tag = self.reverse_primitive_mapping_dict[i]
                    assert layer == 0
                else:
                    _i = i - self.vec_length
                    key1, n_idx = self.reverse_primitive_mapping_neighbor_dict[_i]
                    sma, layer, tag = key1
                    assert layer > 0

                if layer == 0:
                    if isinstance(tag, tuple):
                        tag1 = self.N_atoms - tag[0] + 1
                        tag2 = self.N_atoms - tag[1] + 1
                        tag  = sorted([tag1,tag2])
                        tag  = tuple(tag)
                    ### Atom
                    else:
                        tag = self.N_atoms - tag + 1
                    key = sma, layer, tag
                    j = self.primitive_mapping_dict[key]
                    b_new[j] = b_old[i]
                else:
                    has_layer = True
                    ### Bond
                    if isinstance(tag, tuple):
                        # tag[0] is the atom itself
                        # tag[1] is always -1
                        _tag = self.N_atoms - tag[0] + 1
                        tag_new = (_tag, -1)
                        tag = tag_new
                    ### Atom
                    else:
                        tag = self.N_atoms - tag + 1
                    key2 = sma, layer, tag
                    i_list = self.primitive_mapping_neighbor_dict[key1]
                    j_list = self.primitive_mapping_neighbor_dict[key2]
                    for _i, _j in zip(i_list, j_list):
                        b_new[_j] = b_old[_i]

            _b = b_new.v
            if _b not in bitvec_list_new:
                sma = self.bitvector_to_smarts(b_new)
                if sma not in smarts_list:
                    bitvec_list_new.add(b)
                    smarts_list.add(sma)
            if b in bitvec_list_alloc_dict:
                bitvec_list_alloc_dict_new[b] = bitvec_list_alloc_dict[b]
                bitvec_list_alloc_dict_new[_b] = bitvec_list_alloc_dict[b]
        
        return bitvec_list_new, bitvec_list_alloc_dict_new


    def and_rows(
        self, 
        max_iter = 4, 
        allocations = list(),
        initial_and = list(),
        generate_smarts = False, 
        verbose = False,
        max_neighbor = 3,
        max_on = 1.,
        cleanSmarts = True,
        duplicate_removal = False
        ):
        
        from . import arrays

        max_on = float(max_on)
            
        if max_neighbor > self._max_neighbor:
            max_neighbor = self._max_neighbor
        assert max_neighbor > -1
        
        maxbits = self.maxbits
        
        if len(allocations) == 0:
            allocations = list(range(self.N_allocs))

        initial_bitvec_list, bitvec_list_alloc_dict = self.prepare_bitvectors(
            allocations,
            verbose,
            max_neighbor,
            max_on = 1.,
        )
        
        if verbose:
            print(
                f"Found {len(initial_bitvec_list)} unique bit vectors in initial iteration."
            )

        bitvec_list = set()
        for b in initial_bitvec_list:
            _b = arrays.bitvec(b, maxbits)
            on = len(_b.on())/float(maxbits)
            if on < max_on:
                for b_f in initial_and:
                    b = b_f & b
                bitvec_list.add(b)
        bitvec_list = and_bitvectors(
            bitvec_list,
            max_iter,
            maxbits, 
            max_on=max_on,
            N_atoms=self.N_atoms,
            vec_length=self.vec_length,
            primitive_mapping_dict=self.primitive_mapping_dict,
            primitive_mapping_neighbor_dict=self.primitive_mapping_neighbor_dict,
            reverse_primitive_mapping_dict=self.reverse_primitive_mapping_dict,
            reverse_primitive_mapping_neighbor_dict=self.reverse_primitive_mapping_neighbor_dict,
            verbose=verbose,
        )
        
        if duplicate_removal:
            if verbose:
                print("Removing symmetry duplicates ...")
            bitvec_list, bitvec_list_alloc_dict = self._eliminate_duplicates(
                bitvec_list, 
                bitvec_list_alloc_dict, 
                maxbits,
            )
            if verbose:
                print(
                    f"Found {len(bitvec_list)} unique bit vectors after duplicate removal."
                )

        if verbose:
            print(
                "Generating partitions ..."
            )
        bitvec_dict = dict()
        alloc_dict  = dict()
        smarts_dict = dict()
        on_dict     = dict()
        subset_dict = dict()
        t = 0
        for b in bitvec_list:
            _b = arrays.bitvec(b, maxbits=maxbits)
            on_dict[t] = len(_b.on())/float(maxbits)
            bitvec_dict[t] = b
            alloc_dict[t] = set()
            subset_dict[t] = set()
            if generate_smarts:
                sma = self.bitvector_to_smarts(
                    _b, 
                    cleanSmarts=cleanSmarts
                    )
                smarts_dict[t] = sma
            for a in bitvec_list_alloc_dict:
                ### if b is a subset of a
                if b == (b & a):
                    alloc_dict[t].update(
                        set(
                            bitvec_list_alloc_dict[a]
                        )
                    )
            for _t, _b in enumerate(initial_bitvec_list):
                ### if _b is subset of b
                if b == (_b & b):
                    subset_dict[t].add(_t)
            alloc_dict[t] = tuple(alloc_dict[t])
            t += 1

        return alloc_dict, smarts_dict, on_dict, subset_dict, bitvec_dict


    def bitvector_to_allocations(
        self, 
        b,
        max_neighbor = 3,
        verbose = False,
        ):

        """
        Computing all known allocations for this bitvector.
        An allocation is valid, if its bitvector is superset of
        the query bitvector b. Note that we don't check that the
        bitlength match or the max_neighbor number matches.
        """
        
        if max_neighbor > self._max_neighbor:
            max_neighbor = self._max_neighbor
        assert max_neighbor > -1

        _, bitvec_list_alloc_dict = self.prepare_bitvectors(
            verbose=verbose,
            max_neighbor=max_neighbor,
            max_on=1.,
        )

        allocations = set()
        for _b in bitvec_list_alloc_dict:
            ### b is subset of _b
            ### _b is supserset of b
            if b == (b & _b):
                allocations.update(
                    bitvec_list_alloc_dict[_b]
                    )
        return allocations


    def bitvector_hierarchy_to_allocations(
        self, 
        bitvec_list,
        query_allocations = list(),
        query_systems = list(),
        max_neighbor = 3,
        verbose = False,
        ):

        """
        Transform a bitvector list (assumed to be a SMIRNOFF type hierarchy)
        to an allocation vector.
        """
        
        if max_neighbor > self._max_neighbor:
            max_neighbor = self._max_neighbor
        assert max_neighbor > -1

        _, bitvec_list_alloc_dict = self.prepare_bitvectors(
            verbose=verbose,
            max_neighbor=max_neighbor,
            max_on=1.,
        )

        if len(query_systems) == 0 and len(query_allocations) == 0:
            allocations = [0]*self.N_allocs
            query_allocations = list(range(self.N_allocs))
        elif len(query_systems) > 0:
            query_allocations = list()
            for sys_idx in query_systems:
                valids = np.where(self.sysidx_list == sys_idx)
                ranks  = np.unique(self.rank_list[valids])
                query_allocations += ranks.tolist()
            allocations = [0]*len(query_allocations)
        elif len(query_allocations) > 0:
            allocations = [0]*len(query_allocations)
        for i, b in enumerate(bitvec_list):
            for _b in bitvec_list_alloc_dict:
                if b == (b & _b):
                    for a in bitvec_list_alloc_dict[_b]:
                        if a in query_allocations:
                            allocations[a] = i
        return allocations


    def bitvector_to_smarts(
        self, 
        b, 
        cleanSmarts=True, 
        verbose=False):

        from rdkit import Chem
        from . import arrays

        if isinstance(b, int):
            b = arrays.bitvec(b, self.maxbits)
        
        atom_smarts_dict = dict()
        bond_smarts_dict = dict()
        if verbose:
            print("bit tag layer n_idx SMARTS")
            print("----------------------")
        for i in range(self.maxbits):
            if not b[i]:
                continue
            _i = i
            if i < self.vec_length:
                sma, layer, tag = self.reverse_primitive_mapping_dict[i]
                assert layer == 0
            else:
                i -= self.vec_length
                key, n_idx = self.reverse_primitive_mapping_neighbor_dict[i]
                sma, layer, tag = key
                assert layer > 0   
            if layer == 0:
                key = tag, 0, -1
                ### Bond
                if isinstance(tag, (list,tuple)):
                    if key in bond_smarts_dict:
                        bond_smarts_dict[key].append(sma)
                    else:
                        bond_smarts_dict[key] = [sma]                            
                ### Atom
                else:
                    if key in atom_smarts_dict:
                        atom_smarts_dict[key].append(sma)
                    else:
                        atom_smarts_dict[key] = [sma]
            else:
                key = tag, layer, n_idx
                ### Bond
                if isinstance(tag, (list,tuple)):
                    if key in bond_smarts_dict:
                        bond_smarts_dict[key].append(sma)
                    else:
                        bond_smarts_dict[key] = [sma]                            
                ### Atom
                else:
                    if key in atom_smarts_dict:
                        atom_smarts_dict[key].append(sma)
                    else:
                        atom_smarts_dict[key] = [sma]
                        
            if verbose:
                print(_i, *key, sma)
        if verbose:
            print()
        rdmol = Chem.RWMol(self.base_query_rdmol)
        neighbor_idx_dict = dict()
        for key in atom_smarts_dict:
            tag, layer, n_idx = key
            if layer == 0:
                sma = f"[{'&'.join(atom_smarts_dict[key])}:{tag}]"
                if cleanSmarts:
                    sma = sanitize_atom_smarts(sma)
                atom_idx = self.tagged_atom_dict[tag]
                rdmol.ReplaceAtom(
                    atom_idx,
                    Chem.AtomFromSmarts(sma)
                )
            else:
                sma = f"[{'&'.join(atom_smarts_dict[key])}]"
                if cleanSmarts:
                    sma = sanitize_atom_smarts(sma)
                if "#1" in sma:
                    atom_idx = self.tagged_atom_dict[tag]
                    atom = rdmol.GetAtomWithIdx(atom_idx)
                    if cleanSmarts:
                        if "H" in atom.GetSmarts():
                            key = (tag,-1), layer, n_idx
                            if key in bond_smarts_dict:
                                del bond_smarts_dict[key]
                            continue
                atom_idx = rdmol.AddAtom(
                    Chem.AtomFromSmarts(sma)
                )
                key = (tag,-1), layer, n_idx
                neighbor_idx_dict[key] = atom_idx
                if key not in bond_smarts_dict:
                    bond_smarts_dict[key] = ["~"]                            

        for key in bond_smarts_dict:
            tag, layer, n_idx = key
            if layer == 0:
                sma = '&'.join(bond_smarts_dict[key])
                atom_idx0 = self.tagged_atom_dict[tag[0]]
                atom_idx1 = self.tagged_atom_dict[tag[1]]
                bondidx = rdmol.GetBondBetweenAtoms(
                    atom_idx0,
                    atom_idx1
                ).GetIdx()
                rdmol.ReplaceBond(
                    bondidx,
                    Chem.BondFromSmarts(sma)
                )
            else:
                sma = '&'.join(bond_smarts_dict[key])
                atom_idx0 = self.tagged_atom_dict[tag[0]]
                if key in neighbor_idx_dict:
                    atom_idx1 = neighbor_idx_dict[key]
                else:
                    atom_idx1 = rdmol.AddAtom(
                        Chem.AtomFromSmarts("[*]")
                    )
                    neighbor_idx_dict[key] = atom_idx1
                bond_count = rdmol.AddBond(
                    atom_idx0,
                    atom_idx1,
                )
                rdmol.ReplaceBond(
                    bond_count - 1,
                    Chem.BondFromSmarts(sma)
                )

        sma = Chem.MolToSmarts(
            rdmol.GetMol()
        )
        if cleanSmarts:
            sma = sma.replace("&","")
            sma = sma.replace("!@[#1", "[#1")
            sma = sma.replace("-!@[#1", "-[#1")
        return sma


def smarts_hierarchy_to_allocations(pvec, smarts_list):
    
    from rdkit import Chem
    import numpy as np
    
    N_atoms     = pvec.parameter_manager.N_atoms
    N_systems   = len(pvec.parameter_manager.rdmol_list)
    N_allocs    = pvec.allocations.size
    allocations = pvec.allocations[:]
    rdmol_list  = pvec.parameter_manager.rdmol_list
    atom_list   = np.array(pvec.parameter_manager.atom_list, dtype=int)
    rank_list   = np.array(pvec.parameter_manager.force_ranks, dtype=int)
    sysidx_list = np.array(pvec.parameter_manager.system_idx_list, dtype=int)
    
    dtype = np.dtype(
        'S{:d}'.format(
            N_atoms * atom_list.dtype.itemsize
        )
    )

    atom_list_b = np.frombuffer(
        atom_list.tobytes(), 
        dtype=dtype
    )

    allocations[:] = -1
    for sma_idx, sma in enumerate(smarts_list):
        rdsma = Chem.MolFromSmarts(sma)
        tag_dict = dict()
        for atom in rdsma.GetAtoms():
            tag = atom.GetAtomMapNum()
            if tag > 0:
                tag_dict[tag] = atom.GetIdx()
        for sys_idx in range(N_systems):
            valid_sysidx = np.where(sysidx_list == sys_idx)
            rdmol = rdmol_list[sys_idx]
            matches = rdmol.GetSubstructMatches(
                rdsma,
                useQueryQueryMatches=False,
                uniquify=False,
            )
            matches = np.array(matches, dtype=int)
            if matches.size == 0:
                continue

            atom_list = np.zeros(
                (
                    matches.shape[0],
                    N_atoms
                ), 
                dtype=int
            )
            for tag in tag_dict:
                idx = tag_dict[tag]
                atom_list[:,tag-1] = matches[:,idx]
            atom_list_b1 = np.frombuffer(
                atom_list.tobytes(), 
                dtype=dtype
            )
            atom_list_b2 = np.frombuffer(
                atom_list[:,::-1].tobytes(), 
                dtype=dtype
            )

            atom_list_b12 = np.concatenate((
                atom_list_b1,
                atom_list_b2
            ))

            intersect, c0, c1 = np.intersect1d(
                atom_list_b[valid_sysidx], 
                atom_list_b12,
                return_indices=True
            )
            ranks = np.unique(rank_list[valid_sysidx][c0])
            allocations[ranks] = sma_idx

    return allocations


def get_on_stats(bitvec_list, maxbits=None):
    
    from . import arrays
    import numpy as np

    on_list = list()
    for b in bitvec_list:
        if isinstance(b, int):
            _b = arrays.bitvec(b, maxbits=maxbits)
        elif isinstance(b, arrays.bitvec):
            _b = b
        else:
            raise ValueError(
                f"input {b} has wrong type."
            )
        on_count = len(_b.on())
        on_list.append(
            float(on_count)/float(_b.maxbits)
        )
        
    on_list = np.array(on_list)
    _avg = np.mean(on_list)
    _min = np.min(on_list)
    _max = np.max(on_list)
    
    return _avg, _min, _max   


def bitvec_hierarchy_to_allocations(
    alloc_bitvec_dict, 
    type_bitvec_list,
    allocations_list = None):

    allocations_dict = dict()
    for type_i, b_i in enumerate(type_bitvec_list):
        for b_j in alloc_bitvec_dict:
            if b_i == (b_i & b_j):
                allocs = alloc_bitvec_dict[b_j]
                for a in allocs:
                    allocations_dict[a] = type_i
    if isinstance(allocations_list, list):
        for a in allocations_dict:
            allocations_list[a] = allocations_dict[a]
    return allocations_dict