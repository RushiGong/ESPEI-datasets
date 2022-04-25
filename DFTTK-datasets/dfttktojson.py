"""
Tools for get ESPEI JSON from DFTTK MongoDB
"""

from pymatgen.core import Structure
from dfttk import PRLStructure
from atomate.vasp.database import VaspCalcDb
import os
from dfttk.analysis.formation_energies import get_formation_energy, get_thermal_props
from dfttk.espei_compat import make_dataset, dfttk_config_to_espei, dfttk_occupancies_to_espei
import numpy as np
from dfttk.utils import recursive_flatten
import json
from pymongo import MongoClient

def update_metadata(phase_name, metadata_tag, db_file, collection):
    """
    Return metadata dict of structures with sublattice information

    Parameters
    ----------
    phase_name : str
        phase name of the configuration
    metadata_tag : str
        metadata tag in DFTTK MongoDB of the calculated configuration
    db_file: str
        path of db.json file
    collection: str
        collection name in DFTTK MongoDB

    Return
    ------
    metadata dict: dict
        updated metadata with sublattice information
    """
    vasp_db = VaspCalcDb.from_db_file(db_file, admin=True)
    result_db=vasp_db.db[collection]
    tag_results=result_db.find_one({"metadata.tag": metadata_tag})
    structure=Structure.from_dict(tag_results['structure'])
    ps = PRLStructure.from_structure(structure)
    subl_configuration=ps.sublattice_configuration
    subl_occ=ps.sublattice_occupancies
    subl_ratios=ps.sublattice_site_ratios
    metadata_dict={
    'phase_name': phase_name,
    'tag': metadata_tag,
    'sublattice':{
        'configuration': subl_configuration,
        'occupancies': subl_occ
                }
    }
    return metadata_dict

def dfttk_writeto_json(phase_name, refstate_tags, configuration_to_find, sublattice_site_ratios, db_file, collection, temperature_index, writefile=True):
    """
    Return ESPEI json file writing from DFTTK MongoDB

    Parameters
    ----------
    phase_name : str
        phase name of the configuration
    refstate_tags : str
        metadata tags of reference elements in DFTTK MongoDB
    configuration_to_find: str
        sublattice configuration
    sublattice_site_ratios: str
        sublattice site ratios
    db_file: str
        path of db.json file
    collection: str
        collection name in DFTTK MongoDB
    temperature_index: int
        index of 300 K temperature

    Return
    ------
    JSON file
    """
    vasp_db = VaspCalcDb.from_db_file(db_file, admin=True)
    coll=vasp_db.db[collection]
    refstate_energies = {}
    for el, tag in refstate_tags.items():
        qha_result = coll.find_one({'metadata.tag': tag})
        refstate_energies[el] = get_thermal_props(qha_result)
    configs     = []
    occupancies = []
    hm_values   = []
    sm_values   = []
    cpm_values  = []
    fixed_conds = {'P': 101325, 'T': 0}
    temp_conds = {'P': 101325, 'T': 0}
    for qha_res in coll.find({'metadata.sublattice.configuration': configuration_to_find, 'metadata.phase_name': phase_name}):
        configs.append(qha_res['metadata']['sublattice']['configuration'])
        occupancies.append(qha_res['metadata']['sublattice']['occupancies'])    
        tprops = get_thermal_props(qha_res)
        struct = Structure.from_dict(qha_res['structure'])
        hm_form = get_formation_energy(tprops, struct, refstate_energies, 'HM', idx=temperature_index)
        sm_form = get_formation_energy(tprops, struct, refstate_energies, 'SM', idx=temperature_index)
        cpm_form = get_formation_energy(tprops, struct, refstate_energies, 'CPM', thin=10)[:-2]
        fixed_temp = tprops['T'][temperature_index]
        cpm_temps = tprops['T'][::10][:-2]
        hm_values.append(hm_form)
        sm_values.append(sm_form)
        cpm_values.append(cpm_form)
    fixed_conds['T'] = fixed_temp.tolist()
    temp_conds['T'] = cpm_temps.tolist()

    # make the HM, SM, CPM values arrays of the proper shape
    hm_values = np.array([[hm_values]])
    sm_values = np.array([[sm_values]])
    cpm_values = np.array(cpm_values).T[np.newaxis, ...]

    comps = [c.upper() for c in sorted(recursive_flatten(configuration_to_find))]
    for prop, vals, conds in [('HM_FORM', hm_values, fixed_conds), ('SM_FORM', sm_values, fixed_conds), ('CPM_FORM', cpm_values, temp_conds)]:
        ds = make_dataset(phase_name, prop, sublattice_site_ratios, configs, conds, vals, occupancies=occupancies, tag=tag)
    if writefile is True:
        with open('{}-{}-{}-DFTTK.json'.format('-'.join(comps), phase_name, prop), 'w') as fp:
            json.dump(ds, fp, indent=1)
    else:
        return ds
