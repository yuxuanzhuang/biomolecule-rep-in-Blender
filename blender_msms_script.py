file_top = 'blender_test/test.pdb'

selection = 'protein and not element H'
molecule_name = 'nachrs'
md_frame_start = 0
md_frame_interval = 5
md_frame_end = 100
mdanalysis_dir_location = 'anaconda3/envs/blender/lib/python3.10/site-packages/'
nm_scale = 0.1

# download from https://ccsb.scripps.edu/msms/documentation/
msms_bin = 'msms/msms.x86_64Linux2.2.6.1'

import bpy
from logging import warning
import sys
import site
import numpy as np
import subprocess
import warnings
import itertools
import os
import tempfile
tempdir = tempfile.mkdtemp()

def verify_user_sitepackages(mda_path):
    usersitepackagespath = site.getusersitepackages()

    if os.path.exists(usersitepackagespath) and usersitepackagespath not in sys.path:
            sys.path.append(usersitepackagespath)
    if os.path.exists(mda_path) and mda_path not in sys.path:
            sys.path.append(mda_path)
            
verify_user_sitepackages(mdanalysis_dir_location)

try:
    import MDAnalysis as mda
    from MDAnalysis.coordinates import base
    from MDAnalysis.lib import util
    from MDAnalysis.lib.util import cached
    from MDAnalysis.exceptions import NoDataError
    from MDAnalysis.topology.tables import vdwradii
except:
    warning("Unable to Import MDAnalysis")


class XYZRWriter(base.WriterBase):
    """Writes an XYZR file

    The XYZR file format is used for MSMS generation. It includes the X,Y,Z coordinates
    of the atoms and the vdw radius of the atom.

    .. Links

    .. _MSMS Website:
            https://ccsb.scripps.edu/msms/

    """

    format = 'XYZR'
    multiframe = False
    # these are assumed!
    units = {'time': 'ps', 'length': 'Angstrom'}

    def __init__(self, filename, n_atoms=None, convert_units=True, **kwargs):
        self.filename = filename
        self.n_atoms = n_atoms
        self.convert_units = convert_units

        self._xyzr = util.anyopen(self.filename, 'wt')

    def _get_atoms_elements_or_names(self, atoms):
        """Return a list of atom elements (if present) or fallback to atom names"""
        try:
            return atoms.atoms.elements
        except (AttributeError, NoDataError):
            try:
                return atoms.atoms.names
            except (AttributeError, NoDataError):
                wmsg = ("Input AtomGroup or Universe does not have atom "
                        "elements or names attributes, writer will default "
                        "atom names to 'X'")
                warnings.warn(wmsg)
                return itertools.cycle(('X',))

    def close(self):
        """Close the trajectory file and finalize the writing"""
        if self._xyzr is not None:
            self._xyzr.write("\n")
            self._xyzr.close()
        self._xyzr = None


    def write(self, obj):
        try:
            atoms = obj.atoms
        except AttributeError:
            errmsg = "Input obj is neither an AtomGroup or Universe"
            raise TypeError(errmsg) from None
        else:
            if hasattr(obj, 'universe'):
                # For AtomGroup and children (Residue, ResidueGroup, Segment)
                ts_full = obj.universe.trajectory.ts
                if ts_full.n_atoms == atoms.n_atoms:
                    ts = ts_full
                else:
                    # Only populate a time step with the selected atoms.
                    ts = ts_full.copy_slice(atoms.indices)
            elif hasattr(obj, 'trajectory'):
                # For Universe only --- get everything
                ts = obj.trajectory.ts
            # update atom names
            self.atomnames = self._get_atoms_elements_or_names(atoms)

        self._write_next_frame(ts)


    def _write_next_frame(self, ts=None):
        """
        Write coordinate information in *ts* to the trajectory
        """
        if ts is None:
            if not hasattr(self, 'ts'):
                raise NoDataError('XYZWriter: no coordinate data to write to '
                                    'trajectory file')
            else:
                ts = self.ts

        if self.n_atoms is not None:
            if self.n_atoms != ts.n_atoms:
                raise ValueError('n_atoms keyword was specified indicating '
                                    'that this should be a trajectory of the '
                                    'same model. But the provided TimeStep has a '
                                    'different number ({}) then expected ({})'
                                    ''.format(ts.n_atoms, self.n_atoms))
        else:
            if (not isinstance(self.atomnames, itertools.cycle) and
                len(self.atomnames) != ts.n_atoms):
                self.atomnames = np.array([self.atomnames[0]] * ts.n_atoms)

        if self.convert_units:
            coordinates = self.convert_pos_to_native(
                ts.positions, inplace=False)
        else:
            coordinates = ts.positions

        # Write content
        for atom, (x, y, z) in zip(self.atomnames, coordinates):
            r = vdwradii[atom]
            self._xyzr.write("{0:10.5f} {1:10.5f} {2:10.5f} {3:10.5f}\n"
                            "".format(x, y, z, r))

def write_xyzr_file(atomgroup, location):
    with XYZRWriter(location + '/ag.xyzr') as writer:
        writer.write(atomgroup)
    return location + '/ag.xyzr'

dict_elements = {
    "H":  1,
    "He":  2,
    "Li":  3,
    "Be":  4,
    "B":  5,
    "C":  6,
    "N":  7,
    "O":  8,
    "F":  9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Vac": 104
}


u = mda.Universe(file_top)
ag = u.select_atoms(selection)
output_name = molecule_name

try:
    parent_coll = bpy.data.collections['MolecularNodes']
    parent_coll.name == "MolecularNodes"
    # make the collection active, for creating and disabling new
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[
        'MolecularNodes']
except:
    parent_coll = bpy.data.collections.new('MolecularNodes')
    bpy.context.scene.collection.children.link(parent_coll)
    # make the collection active, for creating and disabling new
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[
        'MolecularNodes']


# create new collection that will house the data, link it to the parent collection
col = bpy.data.collections.new(output_name)
parent_coll.children.link(col)

col_properties = bpy.data.collections.new(output_name + "_properties")
col.children.link(col_properties)


def create_model(name, collection, locations, bonds=[], faces=[]):
    """
    Creates a mesh with the given name in the given collection, from the supplied
    values for the locationso of vertices, and if supplied, bonds and faces.
    """
    # create a new mesh
    atom_mesh = bpy.data.meshes.new(name)
    atom_mesh.from_pydata(locations, bonds, faces)
    new_object = bpy.data.objects.new(name, atom_mesh)
    collection.objects.link(new_object)
    return new_object


def create_properties_model(name, collection, prop_x, prop_y, prop_z, n_atoms):
    """
    Creates a mesh that will act as a look up table for properties about the atoms
    in the actual mesh that will be created.
    """
    if n_atoms == None:
        n_atoms = len(prop_x)

    def get_value(vec, x):
        try:
            return vec[x]
        except:
            return 0

    list_list = list(map(
        lambda x: [
            get_value(prop_x, x),
            get_value(prop_y, x),
            get_value(prop_z, x)
        ], range(n_atoms - 1)
    ))

    create_model(
        name,
        collection,
        list_list
    )


def create_prop_AtomicNum_ChainNum_NameNum(ag, name, collection):
    n = ag.n_atoms

    # get the atomic numbers for the atoms
    try:
        prop_elem_num = list(
            map(lambda x: dict_elements.get(x, 0), ag.elements))

    except:
        try:
            prop_elem = list(
                map(lambda x: mda.topology.guessers.guess_atom_element(x), ag.names))
            prop_elem_num = list(
                map(lambda x: dict_elements.get(x, 0), prop_elem))
        except:
            prop_elem = []
            for i in range(n - 1):
                prop_elem.append(0)

    # get the chain numbers for the atoms
    try:
        prop_chain = ag.chainIDs
        unique_chains = np.array(list(set(prop_chain)))
        prop_chain_num = np.array(
            list(map(lambda x: np.where(x == unique_chains)[0][0], prop_chain)))
    except:
        unique_chains = ["No Chains Found"]
        prop_chain_num = []
        for i in range(n - 1):
            prop_chain_num.append(0)

    # get the name numbers
    try:
        prop_name = ag.names
        unique_names = np.array(list(set(prop_name)))
        prop_name_num = np.array(
            list(map(lambda x: np.where(x == unique_names)[0][0], prop_name)))
    except:
        prop_name = []
        for i in range(n - 1):
            prop_name.append(0)

    create_properties_model(
        name=name,
        collection=collection,
        prop_x=prop_elem_num,
        prop_y=prop_chain_num,
        prop_z=prop_name_num,
        n_atoms=n
    )

    return unique_chains


def create_prop_aaSeqNum_atomNum_atomAAIDNum(ag, name, collection):
    n = ag.n_atoms

    # get the AA sequence numebrs
    try:
        prop_aaSeqNum = ag.resnums
    except:
        prop_aaSeqNum = []
        for i in range(n - 1):
            prop_aaSeqNum.append(0)

    # get the atom indices
    try:
        prop_atomNum = ag.ids
    except:
        prop_atomNum = range(1, n + 1)

    # get the residue names (AA names etc)
    try:
        resnames = ag.resnames
        unique_resnames = np.array(list(set(resnames)))
        prop_aa_ID_num = list(map(
            lambda x: np.where(x == unique_resnames)[0][0], resnames
        ))

    except:
        prop_aa_ID_num = []
        for i in range(n - 1):
            prop_aa_ID_num.append(0)

    create_properties_model(
        name=name,
        collection=collection,
        prop_x=prop_aaSeqNum,
        prop_y=prop_atomNum,
        prop_z=prop_aa_ID_num,
        n_atoms=n
    )


def create_prop_bvalue_isBackbone_isCA(ag, name, collection):
    n = ag.n_atoms

    # setup bvalue properties, likely that most simulations won't actually
    # have bfactors, but here anyway to capture them if they do.
    try:
        prop_bvalue = ag.tempfactors
    except:
        prop_bvalue = []
        for i in range(n - 1):
            prop_bvalue.append(0)

    # setup isBackbone properties, selects backbone for nucleic and protein
    try:
        prop_is_backbone = np.isin(ag.ix, ag.select_atoms(
            'backbone or nucleicbackbone').ix.astype(int))
    except:
        prop_is_backbone = []
        for i in range(n - 1):
            prop_bvalue.append(0)

    try:
        # compare the indices against a subset of indices for only the alpah carbons,
        # convert it to ingeger of 0 = False and 1 = True
        prop_is_CA = np.isin(
            ag.ix, ag.select_atoms("name CA").ix).astype(int)
    except:
        prop_is_CA = []
        for i in range(n - 1):
            prop_is_CA.append(0)

    create_properties_model(
        name=name,
        collection=collection,
        prop_x=prop_bvalue,
        prop_y=prop_is_backbone,
        prop_z=prop_is_CA,
        n_atoms=n
    )


# create the first model, that will be the actual atomic model the user will interact with and display

# for msms
location = tempdir
xyzr_file_loc = write_xyzr_file(ag, location)
process = subprocess.Popen([msms_bin,
                        '-if',
                        xyzr_file_loc,
                        '-of',
                        location + '/msms'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
# print(stdout, stderr)

vert_loc = location + '/msms.vert'
face_loc = location + '/msms.face'

vert_array = np.genfromtxt(vert_loc, skip_header=3)
face_array = np.genfromtxt(face_loc, skip_header=3, dtype=int) - 1

vert_index = vert_array[:, 7].astype(int) - 1

vert_list = list(map(tuple, vert_array[:, :3] * nm_scale))
face_list = list(map(tuple, face_array[:, :3]))

base_model = create_model(
    output_name,
    col,
    vert_list, [], face_list
)


# create the models that will hold the properties associated with each atom
unique_chains = create_prop_AtomicNum_ChainNum_NameNum(
    ag=ag[vert_index],
    name=output_name + "_properties_1",
    collection=col_properties
)
create_prop_aaSeqNum_atomNum_atomAAIDNum(
    ag=ag[vert_index],
    name=output_name + "_properties_2",
    collection=col_properties
)

create_prop_bvalue_isBackbone_isCA(
    ag=ag[vert_index],
    name=output_name + "_properties_3",
    collection=col_properties
)

# hide the created frames collection and the properties collection
bpy.context.layer_collection.children[col.name].children[col_properties.name].exclude = True