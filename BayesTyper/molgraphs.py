#!/usr/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================
import itertools

import numpy as np
from collections import OrderedDict
from rdkit.Chem import AllChem as Chem
import networkx as nx
from openmm import unit


# ==============================================================================
# GLOBAL PARAMETERS
# ==============================================================================
from .constants import (_LENGTH,
                        _ANGLE,
                        _UNIT_QUANTITY,
                        _INACTIVE_GROUP_IDX
                        )

# ==============================================================================
# PRIVATE SUBROUTINES
# ==============================================================================

get_atomic_symbol = Chem.GetPeriodicTable().GetElementSymbol
get_atomic_weight = Chem.GetPeriodicTable().GetAtomicWeight
get_atomic_number = Chem.GetPeriodicTable().GetAtomicNumber


def get_smarts_score(smarts_manager):

    smarts_list = smarts_manager.get_smarts()
    
    length = 0
    for smarts in smarts_list:
        length += len(smarts)
    score = -length
    return score


def _get_smarts_score(smarts_manager):

    from rdkit import Chem
    
    smarts_str   = smarts_manager.get_smarts()
    rdmol_smarts = Chem.MolFromSmarts(smarts_str)

    atom_score_dict = {
        "*"  : 0,
        "#"  : 3,
        "-"  : 3,
        "+"  : 3,
        "H"  : 1,
        "X"  : 1,
        "a"  : 2,
        "A"  : 2,
        "r"  : 2
    }

    bond_score_dict = {
        "~" : 0,
        ":" : 2,
        "@" : 3,
        "-" : 3,
        "=" : 3,
        "#" : 3,
    }

    atom_score = 0
    for key, value in atom_score_dict.items():
        atom_score += atom_score_dict[key] * smarts_str.count(key)

    bond_score = 0
    for key, value in bond_score_dict.items():
        bond_score += bond_score_dict[key] * smarts_str.count(key)

    N_atoms = rdmol_smarts.GetNumAtoms()
    N_bonds = rdmol_smarts.GetNumBonds()

    atom_score /= float(N_atoms)
    bond_score /= float(N_bonds)

    total_score = atom_score + bond_score

    return total_score


class BasePrimitive(object):

    def __init__(self, title):

        self.title = title
        self.symbols_cache_dict = dict()
        
        self.symbols = []
        self.un_orable = []
        self.un_negatable = []
        
        self.max_or_depth = 0


    def get_symbols(
        self, 
        or_depth=1, 
        negate=False, 
        richandor=False):
        
        if or_depth < 0:
            raise ValueError(
                "`or_depth` must be >= 0"
            )    
            
        if or_depth > self.max_or_depth:
            or_depth = self.max_or_depth

        if or_depth in self.symbols_cache_dict:
            return self.symbols_cache_dict[or_depth]

        key = (or_depth, negate)
        if key in self.symbols_cache_dict:
            return self.symbols_cache_dict[key]

        import itertools

        symbols = list()
        for s in self.symbols:
            symbols.append(s)
            if negate:
                if not s in self.un_negatable:
                    symbols.append("!"+s)
            if richandor:
                if not s in self.un_negatable:
                    symbols.append(";"+s)
                    symbols.append(","+s)
                if negate:
                    symbols.append(";!"+s)
                    symbols.append(",!"+s)

        words_list = list()
        for ol in range(0, or_depth+1):
            for word in itertools.combinations(symbols, ol+1):
                word = list(word)
                if len(word) > 1:
                    found_un_or = False
                    found_un_ne = False
                    for un_or in self.un_orable:
                        if un_or in word:
                            found_un_or = True
                            break
                    if found_un_or:
                        continue
                    unique_positive_list = list()
                    for w in word:
                        w = w.replace("!","")
                        if w in unique_positive_list:
                            found_un_ne = True
                            break
                        else:
                            unique_positive_list.append(w)
                    if found_un_ne:
                        continue
                while "" in word:
                    word.remove("")
                if len(word) == 0:
                    word = ""
                elif len(word) == 1:
                    word = f"{word[0]}"
                #elif len(word) > 1:
                else:
                    word = ",".join(word) + ";"
                word_test = word.replace(",","").replace(";","")
                if len(word_test) > 0:
                    words_list.append(word)
                else:
                    words_list.append("")
        self.symbols_cache_dict[key] = tuple(words_list)
        return self.symbols_cache_dict[key]


class ElementPrimitive(BasePrimitive):
    
    def __init__(self):
        
        super().__init__('element')
        
        self.symbols = [
            '*',
            '#1',
            '#6',
            '#7',
            '#8',
        ]
        
        self.un_orable = ["*"]
        self.un_negatable = ["*"]
        
        self.max_or_depth = 1


class AromaticPrimitive(BasePrimitive):
    
    def __init__(self):
        
        super().__init__('aromatic')
        
        self.symbols = [
            '',
            'a',
            'A',
        ]
        
        self.un_orable = ['']
        self.un_negatable = ['a', 'A', '']
        
        self.max_or_depth = 0
        
        
class ChargePrimitive(BasePrimitive):
    
    def __init__(self):
        
        super().__init__('charge')
        
        self.symbols = [
            '',
            '+0',
            '+1',
            '-1',
        ]

        self.un_orable = ['']
        self.un_negatable = ['']

        self.max_or_depth = 0
        
        
class RingPrimitive(BasePrimitive):
    
    def __init__(self):
        
        super().__init__('ring-size')
        
        self.symbols = [
            '',
            'r3',
            'r4',
            'r5',
            'r6',
        ]

        self.un_orable = ['']
        self.un_negatable = ['']

        self.max_or_depth = 0
        
        
class HydrogenPrimitive(BasePrimitive):
    
    def __init__(self):
        
        super().__init__('hydrogen')
        
        self.symbols = [
            '',
            'H0',
            'H1',
            'H2',
            'H3',
        ]

        self.un_orable = ['']
        self.un_negatable = ['']

        self.max_or_depth = 0
        
        
class ConnectivityPrimitive(BasePrimitive):
    
    def __init__(self):
        
        super().__init__('connectivity')
        
        self.symbols = [
            '',
            'X1',
            'X2',
            'X3',
        ]

        self.un_orable = ['']
        self.un_negatable = ['']

        self.max_or_depth = 0


class BondtypePrimitive(BasePrimitive):
    
    def __init__(self):
        
        super().__init__('bond-type')
        
        self.symbols = [
            '~',
            '-',
            '=',
            '#',
        ]

        self.un_orable = ['~']
        self.un_negatable = ['~']

        self.max_or_depth = 0


class BondringPrimitive(BasePrimitive):
    
    def __init__(self):

        super().__init__('bond-in-ring')

        self.symbols = [
            '',
            '@',
        ]

        self.un_orable = ['']
        self.un_negatable = ['', '@']

        self.max_or_depth = 0


class BondaromaticPrimitive(BasePrimitive):
    
    def __init__(self):
        
        super().__init__('bond-aromatic')
        
        self.symbols = [
            '',
            ':',
        ]

        self.un_orable = ['']
        self.un_negatable = ['', ':']

        self.max_or_depth = 0


class SmartsManager(object):

    def __init__(
        self,
        title,
        tag_list,
        atom_decorator_list = None,
        bond_decorator_list = None,
        atom_decorator_andlength = 2,
        bond_decorator_andlength = 1,
        atom_env_orlength = 1,
        bond_env_orlength = 1,
        atom_env_layers = 1,
        use_compressed_allocations = False,
        use_rich_logic = False,
        use_negate_logic = False,
        element_primitive_orlen=0,
        bondtype_primitive_orlen=0,
        rdmol_list = list()
    ):
        
        """
        If we set `use_compressed_allocations=True`, all features
        minues the ones already activated will be accessible to
        each element in the allocations vector. This makes the vector
        alot shorter but will be more degenerate (i.e. multiple realizations
        of the vector lead to the same smarts. In this mode, `atom_decorator_andlength`
        and `bond_decorator_andlength` will determine how many primitives can be
        applied to a given atom.
        If `use_compressed_allocations=False`, then each element in the allocations vector
        encodes exactly one primitive. In this case `atom_decorator_andlength` and 
        `bond_decorator_andlength` are set both to `1`.
        If `use_rich_logic=True` we are able to sample each primitive with a `&`,`;` or `,`.
        """

        self.title = title
        
        tagged_smarts_dict = dict()
        for tag in tag_list:
            tagged_smarts_dict[tag] = dict()
            if len(tag) == 1:
                tagged_smarts_dict[(f"{tag[0]}-l",)] = dict()

        if atom_decorator_list == None:
            atom_decorator_list = [
                AromaticPrimitive(),
                ChargePrimitive(),
                RingPrimitive(),
                HydrogenPrimitive(),
                ConnectivityPrimitive()
            ]

        if bond_decorator_list == None:
            bond_decorator_list = [
                BondringPrimitive(),
                BondaromaticPrimitive()
            ]
            
        if not use_compressed_allocations:
            atom_decorator_andlength = 1
            bond_decorator_andlength = 1

        from BayesTyper.vectors import BaseVector
        self.allocations = BaseVector(dtype=int)

        elementprimitive = ElementPrimitive()
        bondprimitive    = BondtypePrimitive()
        elementprimitive.max_or_depth = element_primitive_orlen
        bondprimitive.max_or_depth = bondtype_primitive_orlen
        for tag in tagged_smarts_dict:
            if len(tag) == 1:
                if tag[0].endswith('-l'):
                    for i in range(atom_env_layers):
                        tagged_smarts_dict[tag][i] = [[elementprimitive]]
                        self.allocations.append(_INACTIVE_GROUP_IDX)
                        for _ in range(atom_decorator_andlength):
                            if use_compressed_allocations:
                                tagged_smarts_dict[tag][i].append(atom_decorator_list)
                                self.allocations.append(_INACTIVE_GROUP_IDX)
                            else:
                                for decorator in atom_decorator_list:
                                    tagged_smarts_dict[tag][i].append([decorator])
                                    self.allocations.append(_INACTIVE_GROUP_IDX)

                else:
                    for i in range(atom_env_orlength):
                        tagged_smarts_dict[tag][i] = [[elementprimitive]]
                        self.allocations.append(_INACTIVE_GROUP_IDX)
                        for _ in range(atom_decorator_andlength):
                            if use_compressed_allocations:
                                tagged_smarts_dict[tag][i].append(atom_decorator_list)
                                self.allocations.append(_INACTIVE_GROUP_IDX)
                            else:
                                for decorator in atom_decorator_list:
                                    tagged_smarts_dict[tag][i].append([decorator])
                                    self.allocations.append(_INACTIVE_GROUP_IDX)

            elif len(tag) == 2:
                for i in range(bond_env_orlength):
                    tagged_smarts_dict[tag][i] = [[bondprimitive]]
                    self.allocations.append(_INACTIVE_GROUP_IDX)
                    for _ in range(bond_decorator_andlength):
                        if use_compressed_allocations:
                            tagged_smarts_dict[tag][i].append(bond_decorator_list)
                            self.allocations.append(_INACTIVE_GROUP_IDX)
                        else:
                            for decorator in bond_decorator_list:
                                tagged_smarts_dict[tag][i].append([decorator])
                                self.allocations.append(_INACTIVE_GROUP_IDX)

        decorator_dict = dict()
        decorator_dict[elementprimitive] = 0
        decorator_dict[bondprimitive] = 0
        for decorator in atom_decorator_list:
            decorator_dict[decorator] = len(decorator_dict)
        for decorator in bond_decorator_list:
            decorator_dict[decorator] = len(decorator_dict)

        self.symbols_dict = dict()
        self.allocation_state = dict()
        counts = 0
        for tag in tagged_smarts_dict:
            self.symbols_dict[tag] = dict()
            for orlayer in tagged_smarts_dict[tag]:
                self.symbols_dict[tag][orlayer] = dict()
                for decorator_list in tagged_smarts_dict[tag][orlayer]:
                    self.symbols_dict[tag][orlayer][counts] = dict()
                    if counts not in self.allocation_state:
                        self.allocation_state[counts] = dict()     
                    n_decorators = len(decorator_list)
                    for decorator in decorator_list:
                        decorator_idx = decorator_dict[decorator]
                        if use_rich_logic:
                            if decorator_idx == 0:
                                symbols = decorator.get_symbols(
                                    0, 
                                    negate = use_negate_logic
                                )
                            else:
                                symbols = decorator.get_symbols(
                                    0, 
                                    negate = use_negate_logic,
                                    richandor=True
                                )
                        else:
                            symbols = decorator.get_symbols(
                                negate = use_negate_logic
                                )

                        from rdkit import Chem
                        default = symbols[0]
                        symbols = symbols[1:]
                        if decorator_idx == 0:
                            if decorator == elementprimitive:
                                rdmol_smarts_list = [Chem.MolFromSmarts(f"[{s}]") for s in symbols]
                            else:
                                rdmol_smarts_list = [Chem.MolFromSmarts(f"[*]{s}[*]") for s in symbols]
                        else:
                            if decorator in atom_decorator_list:
                                rdmol_smarts_list = [Chem.MolFromSmarts(f"[*{s}]") for s in symbols]
                            else:
                                rdmol_smarts_list = [Chem.MolFromSmarts(f"[*]~{s}[*]") for s in symbols]

                        symbols_in = list()
                        for symbol, rdmol_smarts in zip(symbols, rdmol_smarts_list):
                            all_are_matches  = True
                            none_are_matches = True
                            Chem.SanitizeMol(rdmol_smarts)
                            for rdmol in rdmol_list:
                                matches = rdmol.GetSubstructMatches(rdmol_smarts)
                                N_matches = len(matches)
                                if decorator == elementprimitive or decorator in atom_decorator_list:
                                    ref_number = rdmol.GetNumAtoms(onlyExplicit=False)
                                    if ref_number == N_matches:
                                        all_are_matches *= True
                                    else:
                                        all_are_matches *= False
                                    if N_matches == 0:
                                        none_are_matches *= True
                                    else:
                                        none_are_matches *= False
                                if decorator == bondprimitive or decorator in bond_decorator_list:
                                    ref_number = rdmol.GetNumBonds(onlyHeavy=False)
                                    if ref_number == N_matches:
                                        all_are_matches *= True
                                    else:
                                        all_are_matches *= False
                                    if N_matches == 0:
                                        none_are_matches *= True
                                    else:
                                        none_are_matches *= False
                            if all_are_matches == none_are_matches:
                                symbols_in.append(symbol)

                        self.symbols_dict[tag][orlayer][counts][decorator_idx] = default, symbols_in
                        self.allocation_state[counts] = decorator_idx
                    counts += 1


    def get_index_list(self, tag, orlayer, decorator_idx=None):

        import numpy as np

        index_list = list()
        has_decorator_query = True
        if decorator_idx == None:
            has_decorator_query = False
        elif isinstance(decorator_idx, int):
            decorator_idx = [decorator_idx]
        elif isinstance(decorator_idx, list):
            pass
        elif isinstance(decorator_idx, np.ndarray):
            decorator_idx = decorator_idx.tolist()
        else:
            raise ValueError(
                "`decorator_idx` must be of type list, int or None"
            )

        for counts in self.symbols_dict[tag][orlayer]:
            if not has_decorator_query:
                index_list.append(counts)
            else:
                for _decorator_idx in self.symbols_dict[tag][orlayer][counts]:
                    if _decorator_idx in decorator_idx:
                        index_list.append(counts)
        return index_list


    def get_tag(self, index):
        
        found_it = False
        for tag in self.symbols_dict:
            for orlayer in self.symbols_dict[tag]:
                if index in self.symbols_dict[tag][orlayer]:
                    self_tag = tag
                    self_orlayer = orlayer
                    found_it = True
                    break
            if found_it:
                break
        return self_tag, self_orlayer


    def get_available_decorator_idxs(self, index):

        self_tag, self_orlayer = self.get_tag(index)
        index_list = self.get_index_list(self_tag, self_orlayer)
        decorator_idx_list = self.symbols_dict[self_tag][self_orlayer][index].keys()

        exclusion_list = list()
        for _index in index_list:
            if self.allocations[_index] == _INACTIVE_GROUP_IDX:
                continue
            if _index == index:
                continue
            exclusion_list.append(
                self.allocation_state[_index]
            )
        available_decorator_idx_list = list()
        for decorator_idx in decorator_idx_list:
            if not decorator_idx in exclusion_list:
                available_decorator_idx_list.append(decorator_idx)

        available_decorator_idx_list = sorted(available_decorator_idx_list)
        return available_decorator_idx_list


    def get_available_symbols(self, index, asdict=False):

        available_decorator_idx_list = self.get_available_decorator_idxs(index)
        self_tag, self_orlayer = self.get_tag(index)
        if asdict:
            symbol_dict = dict()
        else:
            symbols_list = list()
        for decorator_idx in available_decorator_idx_list:
            default, symbols = self.symbols_dict[self_tag][self_orlayer][index][decorator_idx]
            if asdict:
                symbol_dict.update(
                    {s:decorator_idx for s in symbols}
                )
            else:
                symbols_list.extend(symbols)
        if asdict:
            return symbol_dict
        else:
            return symbols_list


    def get_max_allocation(self, index):

        symbols_list = self.get_available_symbols(index)
        
        return len(symbols_list)


    def set_allocation(self, index, value):

        max_allocation = self.get_max_allocation(index)
        if not value < max_allocation:
            if max_allocation == 0:
                raise ValueError(
                    f"`value` can only be {_INACTIVE_GROUP_IDX}"
                )
            else:
                raise ValueError(
                    f"`value` must be less then {max_allocation}"
                )
        
        symbol_dict_init = dict()
        for tag in self.symbols_dict:
            for orlayer in self.symbols_dict[tag]:
                for counts in self.symbols_dict[tag][orlayer]:
                    if self.allocations[counts] == _INACTIVE_GROUP_IDX:
                        continue
                    if counts == index:
                        continue
                    symbol_list = self.get_available_symbols(counts)
                    _value      = self.allocations[counts]
                    symbol_dict_init[counts] = symbol_list[_value]

        symbol_dict = self.get_available_symbols(index, asdict=True)
        symbol_list = list(symbol_dict.keys())
        if value == _INACTIVE_GROUP_IDX:
            self.allocations[index] = _INACTIVE_GROUP_IDX
        else:    
            symbol      = symbol_list[value]
            decorator_idx = symbol_dict[symbol]
            self.allocations[index] = value
            self.allocation_state[index] = decorator_idx
            
        for tag in self.symbols_dict:
            for orlayer in self.symbols_dict[tag]:
                for counts in self.symbols_dict[tag][orlayer]:
                    if self.allocations[counts] == _INACTIVE_GROUP_IDX:
                        continue
                    if counts == index:
                        continue
                    symbol_list = self.get_available_symbols(counts)
                    symbol = symbol_dict_init[counts]
                    _value  = symbol_list.index(symbol)
                    self.allocations[counts] = _value


    def generate_smarts_dict(self):

        smarts_dict = dict()

        for tag in self.symbols_dict:
            if tag[0].endswith("-l"):
                is_layer_tag = True
            else:
                is_layer_tag = False
            smarts_dict[tag] = ""
            orlayer_smarts_dict = dict()
            for orlayer in self.symbols_dict[tag]:
                orlayer_smarts_dict[orlayer] = "", False
                for counts in self.symbols_dict[tag][orlayer]:
                    a = self.allocations[counts]
                    if a == _INACTIVE_GROUP_IDX:
                        decorator_idx = self.allocation_state[counts]
                        default, symbols = self.symbols_dict[tag][orlayer][counts][decorator_idx]
                        _smarts = default
                        if decorator_idx == 0:
                            orsmarts, isactive = orlayer_smarts_dict[orlayer]
                            orlayer_smarts_dict[orlayer] = orsmarts + _smarts, isactive
                    else:
                        max_allocation = self.get_max_allocation(counts)
                        if not a < max_allocation:
                            raise ValueError(
                                f"allocation at position {counts} is {a} but must be <{max_allocation}"
                            )
                        symbols = self.get_available_symbols(counts)
                        _smarts = symbols[a]
                        orsmarts, _ = orlayer_smarts_dict[orlayer]
                        orlayer_smarts_dict[orlayer] = orsmarts + _smarts, True
            orsmarts_list = list()
            for orlayer in self.symbols_dict[tag]:
                orsmarts, isactive = orlayer_smarts_dict[orlayer]
                if isactive and not orsmarts in orsmarts_list:
                    ### Cosmetics
                    if orsmarts[-1] in [";", ","]:
                        orsmarts = orsmarts[:-1]
                    if orsmarts[0] in [";", ","]:
                        orsmarts = orsmarts[1:]
                    while ";;" in orsmarts:
                        orsmarts.replace(";;","")
                    while ",," in orsmarts:
                        orsmarts.replace(",,","")
                    if orsmarts.startswith("*,"):
                        orsmarts = orsmarts[2:]
                    if orsmarts.endswith(",*"):
                        orsmarts = orsmarts[:-2]
                        
                    orsmarts_list.append(orsmarts)

            if len(orsmarts_list) == 0 and not is_layer_tag:
                orsmarts, _ = orlayer_smarts_dict[0]
                orsmarts_list.append(orsmarts)
            if len(tag) == 1:
                if is_layer_tag:
                    _tag = tag[0].replace("-l", "")
                    n_layer = len(orsmarts_list)
                    for i in range(n_layer):
                        _tag = f"{_tag}-{i}l"
                        smarts_dict[(_tag,)] = "".join(orsmarts_list[i])
                        value = smarts_dict[(_tag,)]
                        smarts_dict[(_tag,)] = f"(~[{value}])"
                else:
                    smarts_dict[tag] += ",".join(orsmarts_list)
                    value = smarts_dict[tag]
                    smarts_dict[tag] = f"[{value}:{tag[0]}]"
            elif len(tag) == 2:
                smarts_dict[tag] += ",".join(orsmarts_list)
            else:
                raise ValueError(
                    f"Tag {tag} not understood."
                )

        return smarts_dict
        
    def get_smarts(self):
        
        pass


class BondSmartsManager(SmartsManager):
    
    def __init__(
        self,
        title,
        **kwargs
        ):
        
        tag_list = [
            ("1",),
            ("2",),
            ("1","2")
        ]

        super().__init__(
            title = title,
            tag_list = tag_list,
            **kwargs
        )
        
    def get_smarts(self):
        
        smarts_dict = self.generate_smarts_dict()

        smarts_str  = smarts_dict[("1",)]
        for tag in smarts_dict:
            if tag[0].startswith("1") and tag[0].endswith("l"):
                smarts_str += smarts_dict[tag]
        smarts_str += smarts_dict[("1","2")]
        smarts_str += smarts_dict[("2",)]
        for tag in smarts_dict:
            if tag[0].startswith("2") and tag[0].endswith("l"):
                smarts_str += smarts_dict[tag]

        return smarts_str
    
    
class AngleSmartsManager(SmartsManager):
    
    def __init__(
        self,
        title,
        **kwargs
        ):
        
        tag_list = [
            ("1",),
            ("2",),
            ("3",),
            ("1","2"),
            ("2","3"),
        ]

        super().__init__(
            title = title,
            tag_list = tag_list,
            **kwargs
        )


    def get_smarts(self):
        
        smarts_dict = self.generate_smarts_dict()

        smarts_str  = smarts_dict[("1",)]
        for tag in smarts_dict:
            if tag[0].startswith("1") and tag[0].endswith("l"):
                smarts_str += smarts_dict[tag]
        smarts_str += smarts_dict[("1","2")]
        smarts_str += smarts_dict[("2",)]
        for tag in smarts_dict:
            if tag[0].startswith("2") and tag[0].endswith("l"):
                smarts_str += smarts_dict[tag]
        smarts_str += smarts_dict[("2","3")]
        smarts_str += smarts_dict[("3",)]
        for tag in smarts_dict:
            if tag[0].startswith("3") and tag[0].endswith("l"):
                smarts_str += smarts_dict[tag]

        return smarts_str


class ProperTorsionSmartsManager(SmartsManager):
    
    def __init__(
        self,
        title,
        **kwargs
        ):
        
        tag_list = [
            ("1",),
            ("2",),
            ("3",),
            ("4",),
            ("1","2"),
            ("2","3"),
            ("3","4"),
        ]

        super().__init__(
            title = title,
            tag_list = tag_list,
            **kwargs
        )

        
    def get_smarts(self):
        
        smarts_dict = self.generate_smarts_dict()

        smarts_str  = smarts_dict[("1",)]
        for tag in smarts_dict:
            if tag[0].startswith("1") and tag[0].endswith("l"):
                smarts_str += smarts_dict[tag]
        smarts_str += smarts_dict[("1","2")]
        smarts_str += smarts_dict[("2",)]
        for tag in smarts_dict:
            if tag[0].startswith("2") and tag[0].endswith("l"):
                smarts_str += smarts_dict[tag]
        smarts_str += smarts_dict[("2","3")]
        smarts_str += smarts_dict[("3",)]
        for tag in smarts_dict:
            if tag[0].startswith("3") and tag[0].endswith("l"):
                smarts_str += smarts_dict[tag]
        smarts_str += smarts_dict[("3","4")]
        smarts_str += smarts_dict[("4",)]
        for tag in smarts_dict:
            if tag[0].startswith("4") and tag[0].endswith("l"):
                smarts_str += smarts_dict[tag]

        return smarts_str


def kronecker(i,j):

    if i == j:
        return 1
    else:
        return 0

def bond_partial(A, B):

    ### See Bakken et al. 10.1063/1.1515483

    ### A: n (0)
    ### B: m (1)

    partial = np.zeros((2, 3), dtype=float)
    AB      = (B-A).in_units_of(_LENGTH)
    length  = np.linalg.norm(AB)
    AB     /= length
    ### nmn                      n m              n n
    partial[0] = AB * (kronecker(0,1) - kronecker(0,0))
    ### mmn                      m m              m n
    partial[1] = AB * (kronecker(1,1) - kronecker(1,0))

    ### Must be unit less
    if type(partial) == unit.quantity.Quantity:
        partial = partial._value

    return partial

def angle_partial(A, B, C):

    ### See Bakken et al. 10.1063/1.1515483

    partial = np.zeros((3, 3), dtype=float)

    ### A: m (0)
    ### B: o (1)
    ### C: n (2)

    ### u: BA (mo)
    ### v: BC (no)

    m = A
    n = C
    o = B

    u  = (m-o).in_units_of(_LENGTH)
    v  = (n-o).in_units_of(_LENGTH)
    length_u = np.linalg.norm(u)
    length_v = np.linalg.norm(v)
    u /= length_u
    v /= length_v
    dot_uv  = np.dot(u,v)
    aux_vec = np.array([1,-1,1], dtype=float)
    if dot_uv < -1.:
        dot_uv = -1.
    elif dot_uv > 1.:
        dot_uv = 1.
    if dot_uv != 0.:
        w = np.cross(u, v)
    else:
        if np.dot(u, aux_vec) and np.dot(v, aux_vec):
            w = np.cross(u, aux_vec)
        else:
            ### Taking care of linear case
            aux_vec = np.array([-1,1,1], dtype=float)
            w = np.cross(u, aux_vec)

    v1 = np.cross(u, w)/length_u
    v2 = np.cross(w, v)/length_v

    ### m (0)
    ### o (1)
    ### n (2)

    ### Partial for A
    ### =============
    ### mmo                       m m              m o
    partial[0]  = v1 * (kronecker(0,0) - kronecker(0,1))
    ### mno                       m n              m o
    partial[0] += v2 * (kronecker(0,2) - kronecker(0,1))

    ### Partial for B
    ### =============
    ### omo                       o m              o o
    partial[1]  = v1 * (kronecker(1,0) - kronecker(1,1))
    ### ono                       o n              o o
    partial[1] += v2 * (kronecker(0,2) - kronecker(1,1))

    ### Partial for C
    ### =============
    ### nmo                       n m              n o
    partial[2]  = v1 * (kronecker(2,0) - kronecker(2,1))
    ### nno                       n n              n o
    partial[2] += v2 * (kronecker(2,2) - kronecker(2,1))

    if type(partial) == unit.quantity.Quantity:
        partial = partial.in_units_of(unit.radian/_LENGTH)
    else:
        partial = partial * unit.radian / _LENGTH

    return partial

def dihedral_partial(A, B, C, D):

    ### See Bakken et al. 10.1063/1.1515483

    partial = np.zeros((4, 3), dtype=float)

    ### A: m (0)
    ### B: o (1)
    ### C: p (2)
    ### D: n (3)

    ### u: BA (om)
    ### v: CD (pn)
    ### w: CB (po)

    m = A
    n = D
    o = B
    p = C

    u  = (m-o).in_units_of(_LENGTH)
    v  = (n-p).in_units_of(_LENGTH)
    w  = (p-o).in_units_of(_LENGTH)
    length_u = np.linalg.norm(u)
    length_v = np.linalg.norm(v)
    length_w = np.linalg.norm(w)
    u /= length_u
    v /= length_v
    w /= length_w

    ### Precompute some terms
    cos_theta_u = np.dot(u,w)
    sin_theta_u = np.sqrt(1. - cos_theta_u**2)
    sin_theta_u2 = sin_theta_u**2

    cos_theta_v = np.dot(v,w)
    sin_theta_v = np.sqrt(1. - cos_theta_v**2)
    sin_theta_v2 = sin_theta_v**2

    cross_uw = np.cross(u,w)
    cross_vw = np.cross(v,w)

    v1 = cross_uw / (length_u * sin_theta_u2)
    v2 = cross_vw / (length_v * sin_theta_v2)
    v3 = cross_uw * cos_theta_u / (length_w * sin_theta_u2)
    v4 = cross_vw * cos_theta_v / (length_w * sin_theta_v2)

    ### A: m (0)
    ### B: o (1)
    ### C: p (2)
    ### D: n (3)

    ### Partial for A
    ### =============
    ### mmo                       m m              m o
    partial[0]  = v1 * (kronecker(0,0) - kronecker(0,1))
    ### mpn                       m p              m n
    partial[0] += v2 * (kronecker(0,2) - kronecker(0,3))
    ### mop                              m o              m p
    partial[0] += (v3 - v4) * (kronecker(0,1) - kronecker(0,2))

    ### Partial for B
    ### =============
    ### omo                       o m              o o
    partial[1]  = v1 * (kronecker(1,0) - kronecker(1,1))
    ### opn                       o p              o n
    partial[1] += v2 * (kronecker(1,2) - kronecker(1,3))
    ### oop                              o o              o p
    partial[1] += (v3 - v4) * (kronecker(1,1) - kronecker(1,2))

    ### Partial for C
    ### =============
    ### pmo                       p m              p o
    partial[2]  = v1 * (kronecker(2,0) - kronecker(2,1))
    ### ppn                       p p              p n
    partial[2] += v2 * (kronecker(2,2) - kronecker(2,3))
    ### pop                              p o              p p
    partial[2] += (v3 - v4) * (kronecker(2,1) - kronecker(2,2))

    ### Partial for D
    ### =============
    ### nmo                       n m              n o
    partial[3]  = v1 * (kronecker(3,0) - kronecker(3,1))
    ### npn                       n p              n n
    partial[3] += v2 * (kronecker(3,2) - kronecker(3,3))
    ### nop                              n o              n p
    partial[3] += (v3 - v4) * (kronecker(3,1) - kronecker(3,2))

    if type(partial) == unit.quantity.Quantity:
        partial = partial.in_units_of(unit.radian/_LENGTH)
    else:
        partial = partial * unit.radian / _LENGTH

    return partial

def pts_to_bond(A, B):
    AB   = (B-A).in_units_of(_LENGTH)
    dist = np.linalg.norm(AB)
    if type(dist) == unit.quantity.Quantity:
        dist = dist.in_units_of(_LENGTH)
    else:
        dist = dist * _LENGTH
    return dist

def pts_to_angle(A, B, C):
    BA  = (A-B).in_units_of(_LENGTH)
    BC  = (C-B).in_units_of(_LENGTH)
    BA /= np.linalg.norm(BA)
    BC /= np.linalg.norm(BC)
    dotABC = np.dot(BA,BC)
    if type(dotABC) == unit.quantity.Quantity:
        dotABC = dotABC._value
    if dotABC < -1.:
        dotABC = -1.
    elif dotABC > 1.:
        dotABC = 1.
    ang = np.arccos(dotABC)
    if type(ang) == unit.quantity.Quantity:
        ang = ang.in_units_of(unit.radian)
    else:
        ang = ang * unit.radian
    return ang

def pts_to_dihedral(A, B, C, D):
    BA  = (A-B).in_units_of(_LENGTH)
    BC  = (C-B).in_units_of(_LENGTH)
    CD  = (C-D).in_units_of(_LENGTH)
    CB  = (C-B).in_units_of(_LENGTH)
    BA /= np.linalg.norm(BA)
    BC /= np.linalg.norm(BC)
    CD /= np.linalg.norm(CD)
    CB /= np.linalg.norm(CB)
    n1  = np.cross(BC,BA)
    n2  = np.cross(CD,CB)
    if type(n1) == unit.quantity.Quantity:
        n1 = n1.in_units_of(_LENGTH)
        n2 = n2.in_units_of(_LENGTH)
    else:
        n1 = n1 * _LENGTH
        n2 = n2 * _LENGTH
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    dotn1n2 = np.dot(n1,n2)
    if type(dotn1n2) == unit.quantity.Quantity:
        dotn1n2 = dotn1n2._value
    if dotn1n2 < -1.:
        dotn1n2 = -1.
    elif dotn1n2 > 1.:
        dotn1n2 = 1.
    dih = np.arccos(dotn1n2)
    if type(dih) == unit.quantity.Quantity:
        dih = dih.in_units_of(unit.radian)
    else:
        dih = dih * unit.radian
    sign = np.dot(np.cross(n1,n2), BC)
    if type(sign) == unit.quantity.Quantity:
        sign = sign._value
    if sign < 0.:
        dih = -dih
    return dih


def add_basin_features(qgraph, basin_dict):

    property_name_list = list(range(basin_dict['integration']['number_of_properties']))
    for prop_dict in basin_dict['integration']['properties']:
        property_name_list[prop_dict['id'] - 1 ] = prop_dict['label']

    ### Then add the new data to the nodes in the already build graph
    for attr_dict in basin_dict['integration']['attractors']:
        for idx, integral_val in enumerate(attr_dict['integrals']):
            if isinstance(integral_val, np.ndarray):
                integral_val_flat = integral_val.flatten()
                for i in range(integral_val_flat.size):
                    name = f"{property_name_list[idx]}_{i}"
                    val  = integral_val_flat[i]
                    qgraph.nodes[attr_dict['id']-1][name] = val
            elif isinstance(integral_val, float):
                name = property_name_list[idx]
                val  = integral_val
                qgraph.nodes[attr_dict['id']-1][name] = val
            elif isinstance(integral_val, int):
                name = property_name_list[idx]
                val  = integral_val
                qgraph.nodes[attr_dict['id']-1][name] = float(val)
            else:
                raise ValueError(f"Value {integral_val} not understood.")


def get_qgraph(qdict):

    qgraph  = nx.Graph()
    N_atoms = qdict['structure']['number_of_nonequivalent_atoms']
    N_cps   = qdict['critical_points']['number_of_nonequivalent_cps']
    N_spec  = qdict['structure']['number_of_species']

    ### 1.) Restructure the data a bit into these dicts and np.ndarrays
    ### ===============================================================
    # Contains encoding for each atomic species
    species_dict = OrderedDict()
    # Contains information about the actual atoms in the system. Mainly
    # species and coordinates.
    atom_dict    = OrderedDict()
    # Contains imformation about the critical points. Mainly signature,
    # rank and spatially integrated properties
    bcp_dict     = OrderedDict()
    # Coordinates of the atoms and bcps
    atom_crds   = np.zeros((N_atoms, 3), dtype=float)
    bcp_crds    = np.zeros((N_cps, 3), dtype=float)
    for i in range(N_spec):
        species_idx               = qdict['structure']['species'][i]['id']
        species_dict[species_idx] = qdict['structure']['species'][i]['atomic_number']
    for i in range(N_atoms):
        at_idx              = qdict['structure']['nonequivalent_atoms'][i]['id']
        atom_dict[at_idx]   = qdict['structure']['nonequivalent_atoms'][i]
        atom_crds[at_idx-1] = qdict['structure']['nonequivalent_atoms'][i]['cartesian_coordinates']
    for i in range(N_cps):
        bcp_idx             = qdict['critical_points']['nonequivalent_cps'][i]['id']
        bcp_dict[bcp_idx]   = qdict['critical_points']['nonequivalent_cps'][i]
        bcp_crds[bcp_idx-1] = qdict['critical_points']['nonequivalent_cps'][i]['cartesian_coordinates']

    ### 2.) Build the adjecency matrix for the molecule. 
    ### ================================================
    ### Instead of having only 0/1 as entries, we use the index of the bcp
    ### connecting two atoms.
    ### Entry '0' still means that there is no bond.
    adj_matrix = np.zeros((N_atoms, N_atoms), dtype=int)
    for i in range(N_cps):
        bcp_idx = i + 1
        ### We only want edges, not nocleus (=nodes)
        if bcp_dict[bcp_idx]['is_nucleus']:
            continue
        ### Make sure we only have bond critical points
        ### and nothing else.
        ### Note: -1 is for bcp, and -3 is for non-nuclear
        ### attractors, e.g. on triple bonds.
        if not bcp_dict[bcp_idx]['signature'] in [-1,-3]:
            continue
        ### if rank is not three, the density is unstable
        ### Is it maybe better to skip the molecule completely?
        if bcp_dict[bcp_idx]['rank'] != 3:
            continue            
        bcp_crd = bcp_crds[i]
        dists   = np.linalg.norm(bcp_crd-atom_crds, axis=1)
        ### Nearest two neighbors of current bcp
        nearest2 = np.argsort(dists)[:2]
        adj_matrix[nearest2[0], nearest2[1]] = bcp_idx
        adj_matrix[nearest2[1], nearest2[0]] = bcp_idx

    ### 3.) Build the graph object from the adjecency matrix
    ### ====================================================
    ### Beginn with the nodes
    for i in range(N_atoms):
        at_idx = i + 1
        qgraph.add_node(i,
                        atomic_num  = species_dict[atom_dict[at_idx]['species']],
                        atomic_crds = atom_dict[at_idx]['cartesian_coordinates'],
                        id          = at_idx,
                        )
    ### Then add the edges
    for i in range(N_atoms):
        for j in range(i+1, N_atoms):
            bcp_idx = adj_matrix[i,j]
            if bcp_idx == 0:
                continue
            k = bcp_idx - 1
            
            hess_eigv = np.sort(bcp_dict[bcp_idx]['hessian_eigenvalues'])
            r1        = np.linalg.norm(atom_crds[i]-bcp_crds[k])
            r2        = np.linalg.norm(atom_crds[j]-bcp_crds[k])
            prop_dict = OrderedDict()

            ### Manually compute some properties
            prop_dict["bcp_ratio"]   = abs(r1-r2)
            prop_dict["ellipticity"] = np.abs(hess_eigv[0]/hess_eigv[1])-1.
            prop_dict["length"]      = np.linalg.norm(atom_crds[i]-atom_crds[j])
            #prop_dict["is_hbond"]    = False
            ### Add all other properties
            for pointprop in bcp_dict[bcp_idx]["pointprops"]:
                if pointprop["name"] == "stress":
                    prop_dict["stress_0"] = pointprop["stress_tensor_eigenvalues"][0]
                    prop_dict["stress_1"] = pointprop["stress_tensor_eigenvalues"][1]
                    prop_dict["stress_2"] = pointprop["stress_tensor_eigenvalues"][2]
                else:
                    prop_dict[pointprop["name"]] = pointprop["value"]
            qgraph.add_edge(i,j,**prop_dict)

    ### Finally, remove hydrogen bonds from graph
    for node in qgraph.nodes(data=True):
        node_idx, node_data = node
        if node_data['atomic_num'] != 1:
            continue
        neighbors = list(qgraph.neighbors(node_idx))
        if len(neighbors) > 1:
            crdsH           = np.array(node_data['atomic_crds'])
            covalent_idx    = -1
            covalent_length = 0.
            ### Find neighbor with shortest bond length
            for node_n_idx in neighbors:
                crdsX  = np.array(qgraph.nodes[node_n_idx]['atomic_crds'])
                length = np.linalg.norm(crdsH-crdsX)
                if length > covalent_length:
                    covalent_idx    = node_n_idx
                    covalent_length = length
            assert covalent_idx != -1
            #qgraph[node_idx][covalent_idx]['is_hbond'] = True
            qgraph.remove_edge(node_idx, covalent_idx)

    return qgraph


class ZMatrix(object):

    ### Note, internally units are nanometer for length and cart coordinates
    ### and radians for angles. However, the output of the zmatrix is degree
    ### instead of radian, since most QC programs use it.

    def __init__(self, rdmol, root_atm_idx=0):

        if not root_atm_idx < rdmol.GetNumAtoms():
            raise ValueError("root_atm_idx must be 0<root_atm_idx<N_atms")

        self.rdmol             = rdmol
        self.ordered_atom_list = [None]*rdmol.GetNumAtoms()
        self.z                 = dict()
        self.N_atms            = 0
        self.rank              = list(Chem.CanonicalRankAtoms(rdmol, breakTies=False))
        self.n_non_deadends    = 0

        self.add_atom(root_atm_idx)
        self.order_atoms(root_atm_idx)
        self.zzit()

    def reorder_vector(self, vector, z_order_in=False):

        ### Reorder input vector from input rdmol/zmat
        ### ordering to zmat/rdmol ordering.
        ### If `z_order_in==True`, the input vector
        ### is assumed to have zmatrix ordering.

        vector_cp = copy.deepcopy(vector)
        for idx in range(self.N_atms):
            if z_order_in:
                a_idx            = self.z2a(idx)
                vector_cp[a_idx] = vector[idx]
            else:
                z_idx            = self.a2z(idx)
                vector_cp[z_idx] = vector[idx]
        return vector_cp

    def build_grad_projection(self, wilson_b, grad_x, as_dict=True):

        __doc__= """
        Compute projection of gradient along zmat coordinates.

        `wilson_b` can be either wilson B matrix or
        dict of that matrix.
        `grad_x` can be either flattend coordinate list (i.e. (N,3) -> (3*N))
        or coordinate list. Note, it must be either
        """

        if isinstance(wilson_b, dict):
            wilson_b_flat = list()
            for z_idx in range(1, self.N_atms):
                atm_idxs = self.z[z_idx]
                if z_idx > 0:
                    wilson_b_row = np.zeros(
                        (self.N_atms, 3), dtype=float
                        )
                    wilson_b_row[atm_idxs[:2]] = wilson_b[z_idx][0]
                    wilson_b_flat.append(
                        wilson_b_row.flatten() * unit.dimensionless
                        )
                if z_idx > 1:
                    wilson_b_row = np.zeros(
                        (self.N_atms, 3), dtype=float
                        )
                    wilson_b_row[atm_idxs[:3]] = wilson_b[z_idx][1]
                    wilson_b_flat.append(
                        wilson_b_row.flatten() * wilson_b[z_idx][1].unit
                        )
                if z_idx > 2:
                    wilson_b_row = np.zeros(
                        (self.N_atms, 3), dtype=float
                        )
                    wilson_b_row[atm_idxs[:4]] = wilson_b[z_idx][2]
                    wilson_b_flat.append(
                        wilson_b_row.flatten() * wilson_b[z_idx][2].unit
                        )

            wilson_b_flat = np.array(wilson_b_flat)
        else:
            wilson_b_flat = wilson_b

        if isinstance(grad_x, _UNIT_QUANTITY):
            HAS_UNIT  = True
            grad_unit = grad_x.unit
        else:
            HAS_UNIT  = False
            grad_unit = 1.

        if grad_x.ndim != 1:
            grad_x_flat = grad_x.flatten()
        else:
            grad_x_flat = grad_x

        length_wilson_b  = 0
        length_wilson_b += self.N_atms * 3 * (self.N_atms - 1)
        length_wilson_b += self.N_atms * 3 * (self.N_atms - 2)
        length_wilson_b += self.N_atms * 3 * (self.N_atms - 3)

        if grad_x_flat.size != (self.N_atms * 3):
            raise ValueError(
                f"length of `grad_x_flat` is {grad_x_flat.size}, but must be {self.N_atms * 3}")

        if wilson_b_flat.shape[1] != (self.N_atms * 3):
            raise ValueError(
                f"shape of `wilson_b_flat` is {wilson_b_flat.size}, but must be {length_wilson_b}")

        if wilson_b_flat.size != (length_wilson_b):
            raise ValueError(
                f"length of `wilson_b_flat` is {wilson_b_flat.size}, but must be {length_wilson_b}")

        ### ============================= ###
        ### THIS IS THE ACTUAL PROJECTION ###
        ### ============================= ###
        ###
        ### This is the Bakken et al. approach.
        ### See 10.1063/1.1515483
        ###
        ### Btw:
        ### ... it gives the same gradient
        ### as with the approach by Peng et al. 
        ### See 10.1002/(SICI)1096-987X(19960115)17:1<49::AID-JCC5>3.0.CO;2-0
        ### u = np.eye(B_flat.shape[1])
        ### G = np.dot(B_flat, np.dot(u, B_flat.T))
        ### G_inv  = np.linalg.pinv(G)
        ### Bugx   = np.dot(B_flat, np.dot(u, grad_x_flat))
        ### grad_q = np.dot(G_inv, Bugx)

        Bt_inv = np.linalg.pinv(wilson_b_flat.T)
        grad_q = np.dot(Bt_inv, grad_x_flat)

        if as_dict:
            grad_q_dict = dict()
            z_counts    = 0
            for z_idx in range(1, self.N_atms):
                grad_q_dict[z_idx] = list()
                if z_idx > 0:
                    grad_q_dict[z_idx].append(
                        grad_q[z_counts] * grad_unit
                        )
                    z_counts += 1
                if z_idx > 1:
                    grad_q_dict[z_idx].append(
                        grad_q[z_counts] * grad_unit
                        )
                    z_counts += 1
                if z_idx > 2:
                    grad_q_dict[z_idx].append(
                        grad_q[z_counts] * grad_unit
                        )
                    z_counts += 1

            return grad_q_dict

        return grad_q

    def build_wilson_b(
        self, 
        crds, 
        as_dict=True):

        wilson_b = dict()

        ### Compute wilson B matrix for transformation.
        ### if `as_dict==False`, the final matrix will be
        ### flattend numpy ndarray. Otherwise will be dict.
        ###
        ### Input cartesian crds must be in order of original
        ### rdkit molecule.

        ### Note: z_idx is in the index in the ordering
        ###       of the zmatrix. This ordering is most
        ###       likely different than the atom ordering
        ###       in the original molecule.
        for z_idx in range(1, self.N_atms):
            atm_idxs = self.z[z_idx]
            wilson_b[z_idx] = list()
            if z_idx > 0:
                wilson_b[z_idx].append(
                    bond_partial(*crds[atm_idxs[:2]])
                    )
            if z_idx > 1:
                wilson_b[z_idx].append(
                    angle_partial(
                        *crds[atm_idxs[:3]])
                    )
            if z_idx > 2:
                wilson_b[z_idx].append(
                    dihedral_partial(*crds[atm_idxs[:4]])
                    )
        if not as_dict:
            wilson_b_flat = list()
            for z_idx in range(1, self.N_atms):
                atm_idxs = self.z[z_idx]
                if z_idx > 0:
                    wilson_b_row = np.zeros(
                        (self.N_atms, 3), dtype=float
                        )
                    wilson_b_row[atm_idxs[:2]] = wilson_b[z_idx][0]
                    ### This partial has no units!
                    wilson_b_flat.append(
                        wilson_b_row.flatten() * unit.dimensionless
                        )
                if z_idx > 1:
                    wilson_b_row = np.zeros(
                        (self.N_atms, 3), dtype=float
                        )
                    wilson_b_row[atm_idxs[:3]] = wilson_b[z_idx][1]
                    wilson_b_flat.append(
                        wilson_b_row.flatten() * wilson_b[z_idx][1].unit
                        )
                if z_idx > 2:
                    wilson_b_row = np.zeros(
                        (self.N_atms, 3), dtype=float
                        )
                    wilson_b_row[atm_idxs[:4]] = wilson_b[z_idx][2]
                    wilson_b_flat.append(
                        wilson_b_row.flatten() * wilson_b[z_idx][2].unit
                        )

            wilson_b_flat = np.array(wilson_b_flat)
            return wilson_b_flat

        return wilson_b

    def z2a(self, z_idx):
        return self.ordered_atom_list[z_idx]

    def a2z(self, atm_idx):
        return self.ordered_atom_list.index(atm_idx)

    def zzit(self):

        self.zz = dict()
        for z_idx, atm_idxs in self.z.items():
            self.zz[z_idx] = [self.a2z(atm_idx) for atm_idx in atm_idxs]
        return True

    def get_neighbor_idxs(self, atm_idx):

        atm            = self.rdmol.GetAtomWithIdx(atm_idx)
        idx_rank_list  = list()
        for atm_nghbr in atm.GetNeighbors():
            idx_rank_list.append([self.rank[atm_nghbr.GetIdx()],
                                   atm_nghbr.GetIdx()])
        for idx_rank in sorted(idx_rank_list, key=lambda idx: idx[0]):
            yield idx_rank[1]

    def get_path_length(self, atm_idx1, atm_idx2, maxlength=100):

        if maxlength < 0:
            raise ValueError("maxlength must >0")

        path_length = -1
        if atm_idx1 == atm_idx2:
            path_length = 0
        elif self.is_neighbor_of(atm_idx1, atm_idx2):
            path_length = 1
        else:
            for k in range(2, maxlength):
                neighbor_list = self.get_k_nearest_neighbors(atm_idx1, k)
                if atm_idx2 in neighbor_list:
                    path_length = k
                    break

        return path_length

    def get_shortest_paths(self, atm_idx1, atm_idx2, query_pool=list(), maxattempts=100):

        if maxattempts < 0:
            raise ValueError("maxattempts must >0")

        if atm_idx1 == atm_idx2:
            shortest_paths = [[atm_idx1]]
        elif self.is_neighbor_of(atm_idx1, atm_idx2):
            shortest_paths = [[atm_idx1, atm_idx2]]
        else:
            shortest_paths = list()
            path_length    = self.get_path_length(atm_idx1, atm_idx2)
            nearest1       = self.get_k_nearest_neighbors(atm_idx1, path_length)
            nearest2       = self.get_k_nearest_neighbors(atm_idx2, path_length)
            intersect      = list(set(nearest1).intersection(set(nearest2)))
            if len(query_pool) > 0:
                intersect = list(set(intersect).intersection(set(query_pool)))
            for intersect_1 in itertools.permutations(intersect, path_length-1):
                if maxattempts < 1:
                    break
                kpaths  = [atm_idx1]
                kpaths += [None]*(path_length-1)
                kpaths += [atm_idx2]
                for l in range(1,path_length):
                    for atm_intersect in intersect_1:
                        l1 = self.get_path_length(atm_intersect, atm_idx1)
                        l2 = self.get_path_length(atm_intersect, atm_idx2)
                        if not (l1 == l and l2 == (path_length-l)):
                            continue
                        if self.is_neighbor_of(atm_intersect, kpaths[l-1]):
                            kpaths[l] = atm_intersect
                maxattempts -= 1
                if kpaths in shortest_paths:
                    continue
                if None in kpaths:
                    continue
                shortest_paths += [kpaths]

        return shortest_paths


    def get_k_nearest_neighbors(self, atm_idx, k=3):

        if k < 0:
            raise ValueError("k must be >0")
        if k == 0:
            neighbor_list = [atm_idx]
        else:
            neighbor_list = list(self.get_neighbor_idxs(atm_idx))
        while (k>1):
            klist = list()
            for atm_nghbr_idx1 in neighbor_list:
                for atm_nghbr_idx2 in list(self.get_neighbor_idxs(atm_nghbr_idx1)):
                    if atm_idx == atm_nghbr_idx2:
                        continue
                    if atm_nghbr_idx1 == atm_nghbr_idx2:
                        continue
                    if atm_nghbr_idx2 in neighbor_list:
                        continue
                    if atm_nghbr_idx2 in klist:
                        continue
                    klist.append(atm_nghbr_idx2)
            neighbor_list += klist
            k -= 1
        return neighbor_list

    def add_atom(self, atm_idx):

        ### Check if we can add atom
        if atm_idx in self.ordered_atom_list:
            return False
        else:
            self.ordered_atom_list[self.N_atms] = atm_idx
            self.N_atms += 1
            if not self.is_dead_end(atm_idx):
                self.n_non_deadends += 1
        ### Build the z matrix
        ### The first three atoms are added 'manually'
        if self.N_atms == 1:
            self.z[0] = [self.ordered_atom_list[0]]
            return True
        elif self.N_atms == 2:
            self.z[1] = [self.ordered_atom_list[1],
                         self.ordered_atom_list[0]]
            return True
        elif self.N_atms == 3:
            if self.get_path_length(self.ordered_atom_list[2],
                                    self.ordered_atom_list[0]) == 2:
                self.z[2] = [self.ordered_atom_list[2],
                             self.ordered_atom_list[1],
                             self.ordered_atom_list[0]]
            else:
                self.z[2] = [self.ordered_atom_list[2],
                             self.ordered_atom_list[0],
                             self.ordered_atom_list[1]]
            return True
        else:
            ### First try to find a chemically identical atom
            ### which alrady has been defined
            for query_atm_idx in self.ordered_atom_list[:self.N_atms-1]:
                if query_atm_idx == None:
                    continue
                if self.rank[query_atm_idx] == self.rank[atm_idx]:
                    idx = self.ordered_atom_list.index(query_atm_idx)
                    if self.is_neighbor_of(query_atm_idx, atm_idx) and idx > 2:
                        self.z[self.N_atms-1] = [atm_idx,
                                                 self.z[idx][1],
                                                 self.z[idx][2],
                                                 self.z[idx][3]]
                        return True
            ### If not, try to find a chemically reasonable path
            for query_atm_idx in self.ordered_atom_list[:self.N_atms-1]:
                if query_atm_idx == None:
                    continue
                if self.get_path_length(atm_idx, query_atm_idx) == 3:
                    zlist               = self.get_shortest_paths(atm_idx,
                                                                  query_atm_idx,
                                                                  self.ordered_atom_list[:self.N_atms])
                    if len(zlist) > 0:
                        self.z[self.N_atms-1] = [atm_idx,
                                                 zlist[0][1],
                                                 zlist[0][2],
                                                 zlist[0][3]]
                        return True
            ### If this didn't work, find another path of length 2
            ### Happens in very small molecules like CH4, where the
            ### the longest path is 2
            for query_atm_idx in self.ordered_atom_list[:self.N_atms-1]:
                if query_atm_idx == None:
                    continue
                if self.get_path_length(atm_idx, query_atm_idx) == 2:
                    zlist               = self.get_shortest_paths(atm_idx,
                                                                  query_atm_idx,
                                                                  self.ordered_atom_list[:self.N_atms])
                    if len(zlist) > 0:
                        for atm_idx_tmp in self.ordered_atom_list[:self.N_atms]:
                            if not atm_idx_tmp in zlist[0]:
                                self.z[self.N_atms-1] = [atm_idx,
                                                         zlist[0][1],
                                                         zlist[0][2],
                                                         atm_idx_tmp]
                                return True
            ### If after all, we still don't have a z matrix entry
            ### for the current atom, just build something that is valid.
            for query_atm_idx in self.ordered_atom_list[:self.N_atms-1]:
                if query_atm_idx == None:
                    continue
                if self.get_path_length(atm_idx, query_atm_idx) == 1:
                    self.z[self.N_atms-1] = list()
                    self.z[self.N_atms-1].append(atm_idx)
                    self.z[self.N_atms-1].append(query_atm_idx)
                    for atm_idx_tmp in self.ordered_atom_list[:self.N_atms]:
                        if not atm_idx_tmp in self.z[self.N_atms-1]:
                            self.z[self.N_atms-1].append(atm_idx_tmp)
                        if len(self.z[self.N_atms-1]) == 4:
                            return True
        return False

    def order_atoms(self, atm_idx):

        add_later = list()
        for atm_nghbr_idx in self.get_neighbor_idxs(atm_idx):
            ### We don't want dead ends in the first
            ### 4 atoms. So add them later!
            if self.is_dead_end(atm_nghbr_idx):
                if self.N_atms < 4:
                    add_later.append(atm_nghbr_idx)
                else:
                    self.add_atom(atm_nghbr_idx)
            elif self.add_atom(atm_nghbr_idx):
                self.order_atoms(atm_nghbr_idx)

        for atm_nghbr_idx in add_later:
            self.add_atom(atm_nghbr_idx)

    def is_dead_end(self, atm_idx):

        if len(list(self.get_neighbor_idxs(atm_idx))) < 2:
            return True
        else:
            return False

    def is_neighbor_of(self, atm_idx1, atm_idx2):

        for atm_nghbr_idx in self.get_neighbor_idxs(atm_idx1):
            if atm_nghbr_idx == atm_idx2:
                return True
        return False

    def build_cart_crds(self, z_crds, virtual_bond=None, virtual_angles=None,
                                      virtual_dihedrals=None, attach_crds=None,
                                      z_order=False):

        __doc__ = """
        Build cartesian coordinates using Natural Extension Reference
        Frame algorithm. If `z_order==True`, then return atoms in order
        of the Zmatrix, otherwise use ordering of the original rdkit
        molecule.
        """
        
        ### We use the Natural Extension Reference Frame algorithm.
        ### See DOI 10.1002/jcc.20237 and 10.1002/jcc.25772
        if virtual_bond == None:
            virtual_bond = 1.*_LENGTH
        if virtual_angles == None:
            virtual_angles = np.array([np.pi/2., np.pi/2.])*unit.radian
        if virtual_dihedrals == None:
            virtual_dihedrals = np.array([np.pi/2., np.pi/2., np.pi/3.])*unit.radian
        if attach_crds == None:
            attach_crds = np.eye(3, dtype=float)*_LENGTH
            attach_crds[:,0] = np.array([1,0,0])*_LENGTH
            attach_crds[:,1] = np.array([0,1,0])*_LENGTH
            attach_crds[:,2] = np.array([1,1,0])*_LENGTH
        atm_idx_check            = np.zeros(len(self.z), dtype=bool)
        atm_idx_check[self.z[0]] = True
        cart_crds                = unit.Quantity(np.zeros((len(self.z),3)), _LENGTH)
        for z_idx in range(len(self.z)):
            atm_idxs = self.z[z_idx]
            if not np.all(atm_idx_check[atm_idxs[1:]]):
                raise Exception(f"Not all atoms for row {z_idx} properly defined.")
            if z_idx == 0:
                A        = attach_crds[:,0].in_units_of(_LENGTH)
                B        = attach_crds[:,1].in_units_of(_LENGTH)
                C        = attach_crds[:,2].in_units_of(_LENGTH)
                bond     = virtual_bond.in_units_of(_LENGTH)
                angle    = virtual_angles[0].in_units_of(unit.radian)
                dihedral = virtual_dihedrals[0].in_units_of(unit.radian)
            elif z_idx == 1:
                A        = attach_crds[:,1].in_units_of(_LENGTH)
                B        = attach_crds[:,2].in_units_of(_LENGTH)
                C        = cart_crds[atm_idxs[1]].in_units_of(_LENGTH)
                bond     = z_crds[z_idx][0].in_units_of(_LENGTH)
                angle    = virtual_angles[1].in_units_of(unit.radian)
                dihedral = virtual_dihedrals[1].in_units_of(unit.radian)
            elif z_idx == 2:
                A        = attach_crds[:,2].in_units_of(_LENGTH)
                B        = cart_crds[atm_idxs[2]].in_units_of(_LENGTH)
                C        = cart_crds[atm_idxs[1]].in_units_of(_LENGTH)
                bond     = z_crds[z_idx][0].in_units_of(_LENGTH)
                angle    = z_crds[z_idx][1].in_units_of(unit.radian)
                dihedral = virtual_dihedrals[2].in_units_of(unit.radian)
            else:
                A        = cart_crds[atm_idxs[3]].in_units_of(_LENGTH)
                B        = cart_crds[atm_idxs[2]].in_units_of(_LENGTH)
                C        = cart_crds[atm_idxs[1]].in_units_of(_LENGTH)
                bond     = z_crds[z_idx][0].in_units_of(_LENGTH)
                angle    = z_crds[z_idx][1].in_units_of(unit.radian)
                dihedral = z_crds[z_idx][2].in_units_of(unit.radian)

            r_cos_angle = np.cos(np.pi-angle._value)*bond
            r_sin_angle = np.sin(np.pi-angle._value)*bond

            cart_crds[atm_idxs[0]][0] = r_cos_angle
            cart_crds[atm_idxs[0]][1] = np.cos(dihedral._value)*r_sin_angle
            cart_crds[atm_idxs[0]][2] = np.sin(dihedral._value)*r_sin_angle
            BC  = C-B
            BC /= np.linalg.norm(BC)
            AB  = B-A
            AB /= np.linalg.norm(AB)
            N   = unit.Quantity(np.cross(AB,BC),_LENGTH)
            N  /= np.linalg.norm(N)
            rot       = np.zeros((3,3), dtype=float)*_LENGTH
            rot[:,0]  = BC
            rot[:,1]  = unit.Quantity(np.cross(N,BC), _LENGTH)
            rot[:,1] /= np.linalg.norm(rot[:,1])
            rot[:,2]  = N
            cart_crds[atm_idxs[0]]     = unit.Quantity(np.dot(rot, cart_crds[atm_idxs[0]]), _LENGTH)
            cart_crds[atm_idxs[0]]    += C
            atm_idx_check[atm_idxs[0]] = True

        _cart_crds = np.zeros((len(self.z),3))*_LENGTH
        if z_order:
            for z_idx in range(len(self.z)):
                atm_idxs = self.z[z_idx]
                _cart_crds[z_idx] = cart_crds[atm_idxs[0]]
            cart_crds = _cart_crds

        return cart_crds

    def build_pretty_zcrds(self, crds):

        __doc__ = """
        Build pretty zmat coordinates expecting that ordering
        of cartesian input coordinates is in ordering
        of original rdkit molecule.

        Resulting string can be used as input for psi4.
        """

        z_crds_dict = self.build_z_crds(crds)
        z_string    = []
        for z_idx, atm_idxs in self.z.items():
            atm      = self.rdmol.GetAtomWithIdx(atm_idxs[0])
            number   = atm.GetAtomicNum()
            element  = get_atomic_symbol(number)
            z_row    = [f"{element} "]
            if z_idx > 0:
                for i, z_idx2 in enumerate(self.zz[z_idx][1:]):
                    if i == 0:
                        value = z_crds_dict[z_idx][i].in_units_of(_LENGTH)
                    else:
                        value = z_crds_dict[z_idx][i].in_units_of(_ANGLE)
                    z_row.append(f"{z_idx2+1} {value._value:6.4f} ")
            z_string.append("".join(z_row))
        return "\n".join(z_string)

    def build_z_crds(self, crds):

        __doc__ = """
        Build zmat coordinates expecting that ordering
        of cartesian input coordinates is in ordering
        of original rdkit molecule.
        """

        z_crds_dict = dict()
        for z_idx, atm_idxs in self.z.items():
            z_crds_dict[z_idx] = list()
            if z_idx == 0:
                z_crds_dict[z_idx].append(crds[atm_idxs[0]].in_units_of(_LENGTH))
            if z_idx > 0:
                dist = pts_to_bond(crds[atm_idxs[0]],
                                   crds[atm_idxs[1]])
                z_crds_dict[z_idx].append(dist)
            if z_idx > 1:
                ang = pts_to_angle(crds[atm_idxs[0]],
                                   crds[atm_idxs[1]],
                                   crds[atm_idxs[2]])
                z_crds_dict[z_idx].append(ang.in_units_of(_ANGLE))
            if z_idx > 2:
                dih = pts_to_dihedral(crds[atm_idxs[0]],
                                      crds[atm_idxs[1]],
                                      crds[atm_idxs[2]],
                                      crds[atm_idxs[3]])
                z_crds_dict[z_idx].append(dih.in_units_of(_ANGLE))
        return z_crds_dict
