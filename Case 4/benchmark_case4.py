import openmc as mc
import numpy as np

temp = 300  # in K

# Materials
# =======================================
hom_mix = mc.Material(name='Homegeneous Material')
hom_mix.add_nuclide('U234', 1.16809e-07, 'ao')
hom_mix.add_nuclide('U235', 1.19358e-05, 'ao')
hom_mix.add_nuclide('U238', 1.10859e-04, 'ao')
hom_mix.add_element('Si', 2.75487e-04, 'ao')
hom_mix.add_element('C', 5.23349e-02, 'ao')
hom_mix.add_nuclide('O16', 2.45823e-04, 'ao')
hom_mix.add_element('B', 1.51250e-08, 'ao')
hom_mix.set_density('atom/b-cm', 5.29791e-02)
hom_mix.add_s_alpha_beta('c_Graphite')
#hom_mix.temperature = temp

# Graphite reflector
reflector = mc.Material(name='Graphite reflector')
reflector.add_element('C', 9.0248e-02, 'ao')
reflector.set_density('atom/b-cm', 9.0248e-02)
reflector.add_s_alpha_beta('c_Graphite')

materials = mc.Materials([hom_mix, reflector])


# Geomitry
# =======================================
cube_surface = mc.model.RectangularParallelepiped(
    -50, 50, -50, 50, -50, 50
)
reflector_surface = mc.model.RectangularParallelepiped(
    -150, 150, -150, 150, -150, 150, boundary_type='vacuum'
)
cube_region = -cube_surface
reflection_region = -reflector_surface & +cube_surface
cube_cell = mc.Cell(name='core', region=cube_region, fill=hom_mix)
cube_cell.temperature = temp
reflector_cell = mc.Cell(name='reflector', region=reflection_region, fill=reflector)
reflector_cell.temperature = temp

geometry = mc.Geometry([cube_cell,reflector_cell])

# Run settings
# =======================================
settings = mc.Settings()
settings.batches = 200
settings.inactive = 30
settings.particles = 10000
settings.temperature = {'method': 'interpolation'}

bounds = [-5, 5, -5, 5, -5, 5]
uniform_dist = mc.stats.Box(bounds[:3], bounds[3:])
settings.source = mc.IndependentSource(space=uniform_dist,)

# In-core scattering epithermal -> thermal
# =======================================
# EnergyFilter looks at the energy of eutrons befor a collision
energy_in_epithermal = mc.EnergyFilter([1.86, 2.0e7])   
# EnergyoutFilter looks at the energy of eutrons after a collision
energyout_thermal = mc.EnergyoutFilter([0.0, 1.86])

"""
The tallies scores that there is:
scatter:     rate of scattering collisions (elastic + inelastic + thermal)
flux:        neutron flux, i.e. total track length travelled per volume
absorption:  rate of neutrons absorbed (captured or causing fission)
fission:     rate of fission reactions
nu-fission:  rate of neutrons produced by fission
total:       rate of all collisions of any type
current:     number of neutrons crossing a surface (needs a surface filter)
heating:     energy deposited by neutrons (eV per source neutron)
"""


#Telley counts
epi_to_thermal_tally = mc.Tally(name='epithermal to thermal scatter')
# Cell filter makes that it looks at the specified cell
core_filter = mc.CellFilter([cube_cell])
# epi_to_thermal_tally.filters read as
# [lool in this cell, at neutron energys before collisions, and agter collisions]
epi_to_thermal_tally.filters = [core_filter, energy_in_epithermal, energyout_thermal]
# Counts the scatter (how many times neutrons bounce off nuclei, 
# per source neutron)
epi_to_thermal_tally.scores = ['scatter']

absorption_tally = mc.Tally(name='absorption')
absorption_tally.scores = ['absorption']

core_abs = mc.Tally(name='core absorption')
core_abs.filters = [core_filter]
core_abs.scores = ['absorption']

# Net core -> reflector current through the 6 faces of the core cube
mesh = mc.RegularMesh()
mesh.dimension = (1, 1, 1)
mesh.lower_left = (-50, -50, -50)
mesh.upper_right = (50, 50, 50)

energy_filter = mc.EnergyFilter([0.0, 1.86, 2.0e7])

current_tally = mc.Tally(name='core-reflector current')
current_tally.filters = [mc.MeshSurfaceFilter(mesh),
                         energy_filter]
current_tally.scores = ['current']

# Core average spectra (epithermal-to-thermal flux ratio)
# =======================================
flux_tally = mc.Tally(name='flux spectrum')
flux_tally.filters = [core_filter, energy_filter]
flux_tally.scores = ['flux']

tallies = mc.Tallies([epi_to_thermal_tally, absorption_tally, core_abs,
                      current_tally, flux_tally])


# Export
# =======================================
materials.export_to_xml()
geometry.export_to_xml()
settings.export_to_xml()
tallies.export_to_xml()


# Run
# =======================================
mc.run()

# Tallies print
# =======================================

sp = mc.StatePoint(f'statepoint.{settings.batches}.h5')

epi_to_thermal_scatter = sp.get_tally(name='epithermal to thermal scatter').mean.flatten()[0]
absorption_rate = sp.get_tally(name='absorption').mean.flatten()[0]
core_abs_rate = sp.get_tally(name='core absorption').mean.flatten()[0]

# vacuum outer boundary -> neutrons are destroyed by absorption (core + reflector) or leakage
gt = sp.global_tallies
leakage_rate = gt['mean'][gt['name'] == b'leakage'][0]
destruction_rate = absorption_rate + leakage_rate

# net leakage core -> reflector = outgoing minus incoming over all 6 faces
cur_df = sp.get_tally(name='core-reflector current').get_pandas_dataframe()
def net_leak(df):
    # mesh filter columns are two-level, e.g. ('mesh 1', 'surf')
    surf_col = df[(f'mesh {mesh.id}', 'surf')]
    out = df[surf_col.str.endswith(' out')]['mean'].sum()
    inn = df[surf_col.str.endswith(' in')]['mean'].sum()
    return out - inn
thermal_df = cur_df[cur_df['energy low [eV]'] == 0.0]
epithermal_df = cur_df[cur_df['energy low [eV]'] == 1.86]
net_leak_thermal = net_leak(thermal_df)
net_leak_epithermal = net_leak(epithermal_df)
net_leakage_total = net_leak_thermal + net_leak_epithermal

flux_df = sp.get_tally(name='flux spectrum').get_pandas_dataframe()
thermal_flux = flux_df['mean'][0]
epithermal_flux = flux_df['mean'][1]

core_destruction = core_abs_rate + net_leakage_total
pct_epi_thermal = epi_to_thermal_scatter / core_destruction * 100
epi_to_thermal_ratio = epithermal_flux / thermal_flux

print(f'In-core scattering epithermal -> thermal: {pct_epi_thermal:.1f}%')
print(f'Core average epithermal-to-thermal ratio over the core: {epi_to_thermal_ratio:.2f}')
print(f'Leakage out of system: {leakage_rate / destruction_rate:.3f}')
print(f'Leakage epithermal (> 1.86 eV): {net_leak_epithermal / core_destruction * 100:.1f}%')
print(f'Leakage thermal (< 1.86 eV): {net_leak_thermal / core_destruction * 100:.1f}%')
print(f'Leakage total (core -> reflector): {net_leakage_total / core_destruction * 100:.1f}%')