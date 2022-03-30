# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 09:27:52 2022

@author: Doug
"""

import numpy as np
import pytest

from pyomo.dae import ContinuousSet, DerivativeVar
import pyomo.environ as pyo
from pyomo.network import Port, Arc
from pyomo.dae.flatten import flatten_dae_components

from idaes.generic_models.properties.core.generic.generic_property import (
    GenericParameterBlock,
)
import idaes.generic_models.unit_models as um
from idaes.core import FlowsheetBlock, UnitModelBlock
import idaes.core.util.scaling as iscale
from idaes.core.solvers import use_idaes_solver_configuration_defaults
import idaes

from idaes.generic_models.properties import iapws95
from idaes.core.util.math import safe_log, smooth_max
from idaes.core.util import get_solver
import idaes.core.util.model_serializer as ms
import idaes.core.util.model_statistics as mstat
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
import idaes.logger as idaeslog

import matplotlib.pyplot as plt
from matplotlib import cm as color_map
import matplotlib.animation as animation
import idaes.core.plugins
from idaes.power_generation.properties.natural_gas_PR import get_prop
from idaes.power_generation.unit_models.soc_submodels import (
    cell_flowsheet, contour_grid_data, comp_enthalpy_expr, 
    comp_entropy_expr, SolidOxideCell)
import idaes.power_generation.unit_models.soc_submodels as soc


use_idaes_solver_configuration_defaults()
idaes.cfg.ipopt["options"]["nlp_scaling_method"] = "user-scaling"
idaes.cfg.ipopt["options"]["linear_solver"] = "ma27"
idaes.cfg.ipopt["options"]["max_iter"] = 100
idaes.cfg.ipopt["options"]["halt_on_ampl_error"] = "yes"
solver = pyo.SolverFactory("ipopt")

time_set = [0]
check_scaling = False

t0 = time_set[0]
#time_set = [0, 600] if dynamic else [0]
#time_set = [60*i for i in range(11)]
zfaces = np.linspace(0, 1, 11).tolist()
xfaces_electrode = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
xfaces_electrolyte = [0.0, 0.25, 0.5, 0.75, 1.0]

fuel_comps = ["H2","H2O","N2"]
fuel_tpb_stoich_dict = {"H2": -0.5, "H2O": 0.5, "N2": 0, "Vac":0.5, "O^2-":-0.5}
oxygen_comps = ["O2","N2"]
oxygen_tpb_stoich_dict = {"O2": -0.25, "N2": 0, "Vac":-0.5, "O^2-":0.5}

def cell_flowsheet_model():
    # function that creates a unit model with cell-level variables for testing
    # subcomponents that require them
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(
        default={
            "dynamic": False,
            "time_set": time_set,
            "time_units": pyo.units.s,
        }
    )
    #time_units = m.fs.time_units
    tset = m.fs.config.time
    znodes = m.fs.znodes = pyo.Set(
        initialize=[
            (zfaces[i] + zfaces[i+1]) / 2.0
            for i in range(len(zfaces)-1)
        ]
    )
    iznodes = m.fs.iznodes = pyo.Set(initialize=range(1, len(znodes) + 1))
    
    m.fs.length_z = pyo.Var(initialize=0.25, units=pyo.units.m)
    m.fs.length_y = pyo.Var(initialize=0.25, units=pyo.units.m)
    
    m.fs.current_density = pyo.Var(tset, iznodes, initialize=0,
                            units=pyo.units.A/pyo.units.m**2)
    
    m.fs.temperature_z = pyo.Var(tset, iznodes, initialize=1000,
                            units=pyo.units.K)
    
        
    m.fs.length_y.fix(0.08)
    m.fs.length_z.fix(0.08)
    m.fs.temperature_z.fix(1000)
    m.fs.current_density.fix(0)
    
    return m

def build_test_utility(block, comp_dict, references=None):
    # Takes a unit model and four dictionaries: references (variables),
    # not_references (variables), constraints, and expressions. They should
    # have the attribute name as a key and the length of the attribute as
    # the value. This function goes through and ensures that all these
    # components exist and are the right length, and furthermore they are no
    # unlisted components of these types
    
        if references is not None:
            for attr in references:
                comp = getattr(block,attr)
                assert comp.is_reference()
            for comp in block.component_data_objects(descend_into=False):
                if comp.is_reference():
                    assert comp in references
        for ctype, sub_dict in comp_dict.items():
            for attr, length in sub_dict.items():
                comp = getattr(block, attr)
                assert len(comp) == length
            for comp in block.component_data_objects(ctype=ctype,
                                                 descend_into=False):
                short_name = comp.name.split("[")[0]
                short_name = short_name.split(".")[-1]
                assert short_name in sub_dict.keys()

class TestTPB(object):
    @pytest.fixture(scope="class")
    def model(self):
        m = cell_flowsheet_model()
        iznodes = m.fs.iznodes
        #time_units = m.fs.time_units
        tset = m.fs.config.time
        comps = m.fs.comps = pyo.Set(initialize=fuel_comps)
        
        m.fs.Dtemp = pyo.Var(tset, iznodes, initialize=0,
                                units=pyo.units.K)
        m.fs.conc_ref = pyo.Var(tset, iznodes, comps, initialize=1,
                                units=pyo.units.mol/pyo.units.m**3)
        m.fs.Dconc = pyo.Var(tset, iznodes, comps, initialize=0,
                                units=pyo.units.mol/pyo.units.m**3)
        m.fs.xflux = pyo.Var(tset, iznodes, comps, initialize=0,
                                units=pyo.units.mol/(pyo.units.s*pyo.units.m**2))
        m.fs.qflux_x0 = pyo.Var(tset, iznodes, initialize=0,
                                units=pyo.units.W/pyo.units.m**2)
        m.fs.fuel_tpb = soc.SocTriplePhaseBoundary(
            default={
                "cv_zfaces": zfaces,
                "length_z": m.fs.length_z,
                "length_y": m.fs.length_y,
                "comp_list": fuel_comps,
                "tpb_stoich_dict": fuel_tpb_stoich_dict,
                "current_density": m.fs.current_density,
                "temperature_z": m.fs.temperature_z,
                "Dtemp": m.fs.Dtemp,
                "qflux_x0": m.fs.qflux_x0,
                "conc_ref": m.fs.conc_ref,
                "Dconc": m.fs.Dconc,
                "xflux": m.fs.xflux,
            }
        )
        m.fs.Dtemp.fix(0)
        m.fs.qflux_x0.fix(0)
        m.fs.conc_ref.fix(1)
        m.fs.Dconc.fix(0)
        
        m.fs.fuel_tpb.exchange_current_log_preexponential_factor\
            .fix(pyo.log(1.375e10))
        m.fs.fuel_tpb.exchange_current_activation_energy.fix(120e3)
        m.fs.fuel_tpb.activation_potential_alpha1.fix(0.5)
        m.fs.fuel_tpb.activation_potential_alpha2.fix(0.5)

        m.fs.fuel_tpb.exchange_current_exponent_comp["H2"].fix(1)
        m.fs.fuel_tpb.exchange_current_exponent_comp["H2O"].fix(1)
        
        
        
        return m
    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self,model):
        tpb = model.fs.fuel_tpb
        nz = 10
        nt = 1
        ncomp = 3
        nreact = 2
        build_test_utility(block=tpb,
           comp_dict={
                pyo.Var: {
                    "temperature_z": nz*nt,
                    "Dtemp": nz*nt,
                    "conc_ref": nz*nt*ncomp,
                    "Dconc": nz*nt*ncomp,
                    "xflux": nz*nt*ncomp,
                    "qflux_x0": nz*nt,
                    "current_density": nz*nt,
                    "length_z": 1,
                    "length_y": 1,
                    "qflux_x1": nz*nt, 
                    "mole_frac_comp": nz*nt*ncomp,
                    "log_mole_frac_comp": nz*nt*ncomp,
                    "activation_potential": nz*nt,
                    "activation_potential_alpha1": 1,
                    "activation_potential_alpha2": 1,
                    "exchange_current_exponent_comp": nreact,
                    "exchange_current_log_preexponential_factor": 1,
                    "exchange_current_activation_energy": 1,
                    },
                pyo.Constraint: {
                    "mole_frac_comp_eqn": nz*nt*ncomp,
                    "log_mole_frac_comp_eqn": nz*nt*ncomp,
                    "xflux_eqn": nz*nt*ncomp,
                    "activation_potential_eqn": nz*nt,
                    "qflux_eqn": nz*nt
                    },
                pyo.Expression: {
                    "temperature": nz*nt,
                    "conc": nz*nt*ncomp,
                    "pressure": nz*nt,
                    "ds_rxn": nz*nt,
                    "dh_rxn": nz*nt,
                    "dg_rxn": nz*nt,
                    "nernst_potential": nz*nt,
                    "log_exchange_current_density": nz*nt,
                    "reaction_rate_per_unit_area": nz*nt,
                    "voltage_drop_total": nz*nt
                    },
            },
            references = [
                "temperature_z",
                "Dtemp",
                "conc_ref",
                "Dconc",
                "xflux",
                "qflux_x0",
                "current_density",
                "length_z",
                "length_y",
                ]
        )
        
    @pytest.mark.unit
    def test_dof(self,model):
        assert degrees_of_freedom(model.fs.fuel_tpb) == 0
        
    @pytest.mark.component
    def test_units(self,model):
        #TODO come back to this later
        return
    
    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialization(self, model):
        model.fs.fuel_tpb.initialize(fix_x0=True)
        
class TestContactResistor(object):
    @pytest.fixture(scope="class")
    def model(self):
        m = cell_flowsheet_model()
        iznodes = m.fs.iznodes
        #time_units = m.fs.time_units
        tset = m.fs.config.time
        m.fs.Dtemp = pyo.Var(tset, iznodes, initialize=0,
                                units=pyo.units.K)
        m.fs.qflux_x0 = pyo.Var(tset, iznodes, initialize=0,
                                units=pyo.units.W/pyo.units.m**2)
        
        m.fs.contact = soc.SocContactResistor(
            default={
                "cv_zfaces": zfaces,
                "length_z": m.fs.length_z,
                "length_y": m.fs.length_y,
                "current_density": m.fs.current_density,
                "temperature_z": m.fs.temperature_z,
                "Dtemp": m.fs.Dtemp,
                "qflux_x0": m.fs.qflux_x0,
            }
        )
        m.fs.Dtemp.fix(0)
        m.fs.qflux_x0.fix(0)
        
        m.fs.contact.log_preexponential_factor.fix(pyo.log(0.46e-4))
        m.fs.contact.thermal_exponent_dividend.fix(0)
        m.fs.contact.contact_fraction.fix(1)
        
        return m
    
    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self,model):
        contact = model.fs.contact
        nz = 10
        nt = 1
        build_test_utility(contact,
            comp_dict = {
                pyo.Var: {
                    "temperature_z": nz*nt,
                    "Dtemp": nz*nt,
                    "qflux_x0": nz*nt,
                    "current_density": nz*nt,
                    "length_z": 1,
                    "length_y": 1,
                    "qflux_x1": nz*nt, 
                    "log_preexponential_factor": 1,
                    "thermal_exponent_dividend": 1,
                    "contact_fraction": 1
                    },
                pyo.Constraint: {
                    "qflux_eqn": nz*nt
                    },
                pyo.Expression: {
                    "temperature": nz*nt,
                    "contact_resistance": nz*nt,
                    "voltage_drop": nz*nt,
                    "joule_heating_flux": nz*nt,
                    },
                },
            references = [
                "temperature_z",
                "Dtemp",
                "qflux_x0",
                "current_density",
                "length_z",
                "length_y",
                ]
        )
        
    @pytest.mark.unit
    def test_dof(self, model):
        assert degrees_of_freedom(model.fs.contact) == 0
        
    @pytest.mark.component
    def test_units(self, model):
        #TODO come back to this later
        return
    
    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialization(self, model):
        model.fs.contact.initialize()
        model.fs.qflux_x0.unfix()
        model.fs.contact.qflux_x1.fix()
        model.fs.contact.initialize(fix_qflux_x0=False)

# Class for the electrolyte and interconnect
class TestConductiveSlab(object):
    @pytest.fixture(scope="class")
    def model(self):
        m = cell_flowsheet_model()
        iznodes = m.fs.iznodes
        #time_units = m.fs.time_units
        tset = m.fs.config.time
        m.fs.Dtemp_x0 = pyo.Var(tset, iznodes, initialize=0,
                                units=pyo.units.K)
        m.fs.qflux_x0 = pyo.Var(tset, iznodes, initialize=0,
                                units=pyo.units.W/pyo.units.m**2)
        m.fs.Dtemp_x1 = pyo.Var(tset, iznodes, initialize=0,
                                units=pyo.units.K)
        m.fs.qflux_x1 = pyo.Var(tset, iznodes, initialize=0,
                                units=pyo.units.W/pyo.units.m**2)
        
        m.fs.slab = soc.SocConductiveSlab(
            default={
                "cv_zfaces": zfaces,
                "cv_xfaces": xfaces_electrolyte,
                "length_z": m.fs.length_z,
                "length_y": m.fs.length_y,
                "current_density": m.fs.current_density,
                "temperature_z": m.fs.temperature_z,
                "Dtemp_x0": m.fs.Dtemp_x0,
                "qflux_x0": m.fs.qflux_x0,
                "Dtemp_x1": m.fs.Dtemp_x1,
                "qflux_x1": m.fs.qflux_x1,
            }
        )
        m.fs.Dtemp_x0.fix(0)
        m.fs.qflux_x0.fix(0)
        
        m.fs.slab.length_x.fix(1.4e-4)
        m.fs.slab.heat_capacity.fix(470)
        m.fs.slab.density.fix(5160)
        m.fs.slab.thermal_conductivity.fix(2.16)
        m.fs.slab.resistivity_log_preexponential_factor\
            .fix(pyo.log(1.07e-4))
        m.fs.slab.resistivity_thermal_exponent_dividend.fix(7237)
        
        return m
    
    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self,model):
        slab = model.fs.slab
        nx = 4
        nz = 10
        nt = 1
        build_test_utility(slab,
           comp_dict={
                pyo.Var: {
                    "temperature_z": nz*nt,
                    "current_density": nz*nt,
                    "length_z": 1,
                    "length_y": 1,
                    "Dtemp_x0": nz*nt,
                    "qflux_x0": nz*nt,
                    "Dtemp_x1": nz*nt,
                    "qflux_x1": nz*nt,
                    "Dtemp": nx*nz*nt,
                    "int_energy_density_solid": nx*nz*nt,
                    "length_x": 1,
                    "resistivity_log_preexponential_factor": 1,
                    "resistivity_thermal_exponent_dividend": 1,
                    "heat_capacity": 1,
                    "density": 1,
                    "thermal_conductivity": 1,
                    },
                pyo.Constraint: {
                    "qflux_x0_eqn": nz*nt,
                    "qflux_x1_eqn": nz*nt,
                    "energy_balance_solid_eqn": nx*nz*nt,
                    'int_energy_density_solid_eqn': nx*nz*nt,
                    },
                pyo.Expression: {
                    "temperature_x0": nz*nt,
                    "temperature": nx*nz*nt,
                    "temperature_x1": nz*nt,
                    "dz": nz,
                    "dx": nx,
                    "node_volume": nx*nz,
                    "zface_area": nx,
                    "xface_area": nz,
                    "current": nz*nt,
                    "dTdx": (nx+1)*nz*nt,
                    "dTdz": (nz+1)*nx*nt,
                    "qxflux": (nx+1)*nz*nt,
                    "qzflux": (nz+1)*nx*nt,
                    "resistivity": nx*nz*nt,
                    "resistance": nx*nz*nt,
                    "voltage_drop": nx*nz*nt,
                    "resistance_total": nz*nt,
                    "voltage_drop_total": nz*nt,
                    "joule_heating": nx*nz*nt,
                    },
                },
            references = [
                "temperature_z",
                "current_density",
                "length_z",
                "length_y",
                "Dtemp_x0",
                "qflux_x0",
                "Dtemp_x1",
                "qflux_x1",
                ],
        )
    
    @pytest.mark.unit
    def test_toggle_internal_energy(self, model):
        slab = model.fs.slab
        nx = 4
        nz = 10
        for ix in range(1,nx+1):
            for iz in range(1,nz+1):
                for t in [0]:
                    assert slab.int_energy_density_solid_eqn[t,ix,iz].active
                    assert not slab.int_energy_density_solid[t,ix,iz].fixed
        slab.deactivate_internal_energy()
        for ix in range(1,nx+1):
            for iz in range(1,nz+1):
                for t in [0]:
                    assert not slab.int_energy_density_solid_eqn[t,ix,iz].active
                    assert slab.int_energy_density_solid[t,ix,iz].fixed
        slab.activate_internal_energy()
        for ix in range(1,nx+1):
            for iz in range(1,nz+1):
                for t in [0]:
                    assert slab.int_energy_density_solid_eqn[t,ix,iz].active
                    assert not slab.int_energy_density_solid[t,ix,iz].fixed
    
    @pytest.mark.unit
    def test_dof(self, model):
        assert degrees_of_freedom(model.fs.slab) == 0
        model.fs.slab.deactivate_internal_energy()
        assert degrees_of_freedom(model.fs.slab) == 0
        model.fs.slab.activate_internal_energy()
        assert degrees_of_freedom(model.fs.slab) == 0
        
    @pytest.mark.component
    def test_units(self, model):
        #TODO come back to this later
        return
    
class TestSolidOxideCell(object):
    @pytest.fixture(scope="class")
    def model(self):
        m = pyo.ConcreteModel()
        m.fs = FlowsheetBlock(
            default={
                "dynamic": False,
                "time_set": time_set,
                "time_units": pyo.units.s,
            }
        )
        m.fs.cell = SolidOxideCell(
                default={
                    "has_holdup": True,
                    "cv_zfaces": zfaces,
                    "cv_xfaces_fuel_electrode": xfaces_electrode,
                    "cv_xfaces_oxygen_electrode": xfaces_electrode,
                    "cv_xfaces_electrolyte": xfaces_electrolyte,
                    "fuel_comps": fuel_comps,
                    "fuel_tpb_stoich_dict": fuel_tpb_stoich_dict,
                    "oxygen_comps": oxygen_comps,
                    "oxygen_tpb_stoich_dict": oxygen_tpb_stoich_dict,
                    "opposite_flow": False
                }
            )
        return m 
        
    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self, model):
        cell = model.fs.cell
        nt = 1
        nz = 10
        self.build_tester(cell,nt,nz)
        
        channels = [cell.fuel_chan, cell.oxygen_chan]
        
        for chan in channels:
            assert (cell.temperature_z
                    is chan.temperature_z.referent)
            assert (cell.length_y
                    is chan.length_y.referent)
            assert (cell.length_z
                    is chan.length_z.referent)
        
        
        contact_resistors = [cell.contact_interconnect_fuel_flow_mesh,
                             cell.contact_interconnect_oxygen_flow_mesh,
                             cell.contact_flow_mesh_fuel_electrode,
                             cell.contact_flow_mesh_oxygen_electrode]
        
        for unit in contact_resistors:
            assert (cell.temperature_z
                    is unit.temperature_z.referent)
            assert (cell.current_density
                    is unit.current_density.referent)
            assert (cell.length_y
                    is unit.length_y.referent)
            assert (cell.length_z
                    is unit.length_z.referent)
            
        assert (cell.fuel_chan.qflux_x0 
                is cell.contact_interconnect_fuel_flow_mesh.qflux_x1.referent)
        assert (cell.fuel_chan.Dtemp_x0 
                is cell.contact_interconnect_fuel_flow_mesh.Dtemp.referent)

        assert (cell.fuel_chan.qflux_x1 
                is cell.contact_flow_mesh_fuel_electrode.qflux_x0.referent)
        assert (cell.fuel_chan.Dtemp_x1 
                is cell.contact_flow_mesh_fuel_electrode.Dtemp.referent)
        
        assert (cell.oxygen_chan.qflux_x1 
                is cell.contact_interconnect_oxygen_flow_mesh.qflux_x0.referent)
        assert (cell.oxygen_chan.Dtemp_x1 
                is cell.contact_interconnect_oxygen_flow_mesh.Dtemp.referent)

        assert (cell.oxygen_chan.qflux_x0 
                is cell.contact_flow_mesh_oxygen_electrode.qflux_x1.referent)
        assert (cell.oxygen_chan.Dtemp_x0 
                is cell.contact_flow_mesh_oxygen_electrode.Dtemp.referent)        
        
        electrodes = [cell.fuel_electrode,
                      cell.oxygen_electrode]
        
        for trode in electrodes:
            assert (cell.temperature_z
                    is trode.temperature_z.referent)
            assert (cell.current_density
                    is trode.current_density.referent)
            assert (cell.length_y
                    is trode.length_y.referent)
            assert (cell.length_z
                    is trode.length_z.referent)
            
        assert (cell.fuel_chan.Dtemp_x1
                is cell.fuel_electrode.Dtemp_x0.referent)
        assert (cell.contact_flow_mesh_fuel_electrode.qflux_x1
                is cell.fuel_electrode.qflux_x0.referent)
        assert (cell.fuel_chan.conc
                is cell.fuel_electrode.conc_ref.referent)
        assert (cell.fuel_chan.Dconc_x1
                is cell.fuel_electrode.Dconc_x0.referent)
        assert (cell.fuel_chan.dcdt
                is cell.fuel_electrode.dconc_refdt.referent)
        assert (cell.fuel_chan.xflux_x1
                is cell.fuel_electrode.xflux_x0.referent)
        
        assert (cell.oxygen_chan.Dtemp_x0
                is cell.oxygen_electrode.Dtemp_x1.referent)
        assert (cell.contact_flow_mesh_oxygen_electrode.qflux_x0
                is cell.oxygen_electrode.qflux_x1.referent)
        assert (cell.oxygen_chan.conc
                is cell.oxygen_electrode.conc_ref.referent)
        assert (cell.oxygen_chan.Dconc_x0
                is cell.oxygen_electrode.Dconc_x1.referent)
        assert (cell.oxygen_chan.dcdt
                is cell.oxygen_electrode.dconc_refdt.referent)
        assert (cell.oxygen_chan.xflux_x0
                is cell.oxygen_electrode.xflux_x1.referent)
        
        tpb_list = [cell.fuel_tpb, cell.oxygen_tpb]
        
        for tpb in tpb_list:
            assert (cell.temperature_z
                    is tpb.temperature_z.referent)
            assert (cell.current_density
                    is tpb.current_density.referent)
            assert (cell.length_y
                    is tpb.length_y.referent)
            assert (cell.length_z
                    is tpb.length_z.referent)    
            
        assert (cell.fuel_tpb.Dtemp.referent
                is cell.fuel_electrode.Dtemp_x1)
        assert (cell.fuel_tpb.qflux_x0.referent
                is cell.fuel_electrode.qflux_x1)
        assert (cell.fuel_tpb.conc_ref.referent
                is cell.fuel_chan.conc)
        assert (cell.fuel_tpb.Dconc.referent
                is cell.fuel_electrode.Dconc_x1)
        
        assert (cell.oxygen_tpb.Dtemp.referent
                is cell.oxygen_electrode.Dtemp_x0)
        assert (cell.oxygen_tpb.qflux_x1.referent
                is cell.oxygen_electrode.qflux_x0)
        assert (cell.oxygen_tpb.conc_ref.referent
                is cell.oxygen_chan.conc)
        assert (cell.oxygen_tpb.Dconc.referent
                is cell.oxygen_electrode.Dconc_x0)
        
        assert (cell.temperature_z
                is cell.electrolyte.temperature_z.referent)
        assert (cell.current_density
                is cell.electrolyte.current_density.referent)
        assert (cell.length_y
                is cell.electrolyte.length_y.referent)
        assert (cell.length_z
                is cell.electrolyte.length_z.referent) 
        assert (cell.fuel_electrode.Dtemp_x1
                is cell.electrolyte.Dtemp_x0.referent)
        assert (cell.oxygen_electrode.Dtemp_x1 
                is cell.electrolyte.Dtemp_x1.referent)
        assert (cell.fuel_tpb.qflux_x1
                is cell.electrolyte.qflux_x0.referent)
        assert (cell.oxygen_tpb.qflux_x0
                is cell.electrolyte.qflux_x1.referent)
        
        
    def build_tester(self,cell,nt,nz):
        build_test_utility(cell,
           comp_dict = {
               pyo.Var: {
                   "current_density": nz*nt,
                   "potential": nt,
                   "temperature_z": nz*nt,
                   "length_z": 1,
                   "length_y": 1,
                   },
               pyo.Constraint: {
                   "mean_temperature_eqn": nz*nt,
                   "potential_eqn": nz*nt,
                   "no_qflux_fuel_interconnect_eqn": nz*nt,
                   "no_qflux_oxygen_interconnect_eqn": nz*nt, 
                   },
               pyo.Expression: {
                   "eta_contact": nz*nt,
                   "eta_ohm": nz*nt,
                   "electrical_work": 1
                   },
               UnitModelBlock: {
                   "fuel_chan": 1,
                   "oxygen_chan": 1,
                   "contact_interconnect_fuel_flow_mesh": 1,
                   "contact_interconnect_oxygen_flow_mesh": 1,
                   "contact_flow_mesh_fuel_electrode": 1,
                   "contact_flow_mesh_oxygen_electrode": 1,
                   "fuel_electrode": 1,
                   "oxygen_electrode": 1,
                   "fuel_tpb": 1,
                   "oxygen_tpb": 1,
                   "electrolyte": 1
                   }
               })