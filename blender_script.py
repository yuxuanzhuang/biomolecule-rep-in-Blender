file_top = 'blender_test/test.pdb'
file_traj ='blender_test/test.xtc'


# visualize protein as tube
selection = 'protein and name CA'
molecule_name = 'nachrs_prot'
rep = 'tube'

# visualize lipid as vdw (uncomment and run the script again)
# selection = 'not protein and not resname HOH SOD CLA and not element H'
# molecule_name = 'lipid'
# rep = 'vdw'

nm_scale = 0.1
md_frame_start = 0
md_frame_interval = 5
md_frame_end = 100
mdanalysis_dir_location = 'anaconda3/envs/blender/lib/python3.10/site-packages/'

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


u = mda.Universe(file_top, file_traj)
if rep == 'vdw':
    ag = u.select_atoms(selection)
elif rep == 'tube':
    # restrain the selection to name CA or nucleic backbone for tube rep
    ag = u.select_atoms(selection).select_atoms('name CA or nucleicbackbone') 

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


# generate tube

# for cartoon rep; currently not working
if False:
    location = tempdir
    
    stride_bin = "src/stride"

    ag.write(location + '/stride.pdb')

    strid_out = open(location + "/stride.out", "w")
    process = subprocess.Popen([stride_bin,
                                location + '/stride.pdb'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()


    ss_out = stdout.decode("utf-8").split('\n')[:-1]

    ss_list = []
    reading=False
    for ind, line in enumerate(ss_out):
        if reading:
            ss_list.append(line[24])
        if "Residue" in line:
            reading=True

             
def create_tube_model(name, collection, coordinates):
    # create the Curve Datablock
    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = 2

    # map coords to spline
    polyline = curveData.splines.new('NURBS')
    polyline.points.add(len(coordinates) - 1)
    for i, coord in enumerate(coordinates):
        x,y,z = coord
        polyline.points[i].co = (x, y, z, 1)

    # create Object
    curveOB = bpy.data.objects.new(name, curveData)
    collection.objects.link(curveOB)
    return curveOB


# join chains
def create_joined_tube_object(ag, name, collection, nm_scale):
    base_models = []
    for ag_chain in ag.split('segment'):
        ca_coords = list(map(tuple, ag_chain.positions * nm_scale))
        base_models.append(create_tube_model(
            name=name,
            collection=collection,
            coordinates=ca_coords,
        ))

    model_c = {}
    model_c["object"] = model_c["active_object"] = base_models[0]
    model_c["selected_objects"] = model_c["selected_editable_objects"] = base_models

    bpy.ops.object.join(model_c)
    return base_models[0]

if rep == 'tube':
    base_model = create_joined_tube_object(
                ag=ag,
                name=output_name,
                collection=col,
                nm_scale=nm_scale)
elif rep == 'vdw':
    base_model = create_model(
                name=output_name,
                collection=col,
                locations=ag.positions * nm_scale
                )



# create the models that will hold the properties associated with each atom
unique_chains = create_prop_AtomicNum_ChainNum_NameNum(
    ag=ag,
    name=output_name + "_properties_1",
    collection=col_properties
)

create_prop_aaSeqNum_atomNum_atomAAIDNum(
    ag=ag,
    name=output_name + "_properties_2",
    collection=col_properties
)

create_prop_bvalue_isBackbone_isCA(
    ag=ag,
    name=output_name + "_properties_3",
    collection=col_properties
)

# create the frames

col_frames = bpy.data.collections.new(output_name + "_frames")
col.children.link(col_frames)
# for each model in the pdb, create a new object and add it to the frames collection
# testing out the addition of points that represent the bfactors. You can then in theory
# use the modulo of the index to be able to pick either the position or the bvalue for
# each frame in the frames collection.

def create_frames(universe, collection, start=1, end=50000, time_step=100, name=output_name, nm_scale=0.1):
    """
    From the given universe, add frames to the given collection from the start till the     end, along the given time
    step for each frame.
    """
    counter = 1
    for ts in universe.trajectory:
        if counter % time_step == 0 and counter > start and counter < end:
            create_model(
                name=output_name + "_frame_" + str(counter),
                collection=collection,
                locations=ag.positions * nm_scale
            )
        counter += 1

def create_tube_frames(universe, collection, start=1, end=50000, time_step=100, name=output_name, nm_scale=0.1):
    counter = 1
    for ts in universe.trajectory:
        if counter % time_step == 0 and counter > start and counter < end:
            base_model = create_joined_tube_object(
                ag=ag,
                name=output_name + "_frame_" + str(counter),
                collection=collection,
                nm_scale=nm_scale
            )
        counter += 1

# create the frames from the given universe, only along the given timesteps
if rep == 'tube':
    create_tube_frames(
        universe=u,
        collection=col_frames,
        start=md_frame_start,
        time_step=md_frame_interval,
        end=md_frame_end,
        name=output_name,
        nm_scale=nm_scale
    )
elif rep == 'vdw':
    create_frames(
        universe=u,
        collection=col_frames,
        start=md_frame_start,
        time_step=md_frame_interval,
        end=md_frame_end,
        name=output_name,
        nm_scale=nm_scale
    )

# hide the created frames collection and the properties collection
bpy.context.layer_collection.children[col.name].children[col_frames.name].exclude = True
bpy.context.layer_collection.children[col.name].children[col_properties.name].exclude = True
