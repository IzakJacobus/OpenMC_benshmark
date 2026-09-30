import openmc as mc
import numpy as np

temp = 1200  # in K

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

materials = mc.Materials([hom_mix])


# Geomitry
# =======================================
cube_surface = mc.model.RectangularParallelepiped(
    -50, 50, -50, 50, -50, 50, boundary_type='reflective'
)
cube_region = -cube_surface
cube_cell = mc.Cell(region=cube_region, fill=hom_mix)
cube_cell.temperature = temp

geometry = mc.Geometry([cube_cell])

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
# thermal cut-off = 1.86 eV, per benchmark definition
# =======================================
# EnergyFilter looks at the energy of eutrons befor a collision
energy_in_epithermal = mc.EnergyFilter([1.86, 2.0e7])   
# EnergyoutFilter looks at the energy of eutrons after a collision
energyout_thermal = mc.EnergyoutFilter([0.0, 1.86])

epi_to_thermal_tally = mc.Tally(name='epithermal to thermal scatter')
epi_to_thermal_tally.filters = [energy_in_epithermal, energyout_thermal]
epi_to_thermal_tally.scores = ['scatter']

absorption_tally = mc.Tally(name='absorption')
absorption_tally.scores = ['absorption']

# Core average spectra (epithermal-to-thermal flux ratio)
# =======================================
flux_energy_filter = mc.EnergyFilter([0.0, 1.86, 2.0e7])
flux_tally = mc.Tally(name='flux spectrum')
flux_tally.filters = [flux_energy_filter]
flux_tally.scores = ['flux']

tallies = mc.Tallies([epi_to_thermal_tally, absorption_tally, flux_tally])


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
# reflective boundaries on all 6 faces (Case 1) -> leakage = 0
destruction_rate = absorption_rate

flux_df = sp.get_tally(name='flux spectrum').get_pandas_dataframe()
thermal_flux = flux_df['mean'][0]
epithermal_flux = flux_df['mean'][1]

pct_epi_thermal = epi_to_thermal_scatter / destruction_rate * 100
epi_to_thermal_ratio = epithermal_flux / thermal_flux

print(f'In-core scattering epithermal -> thermal: {pct_epi_thermal:.1f}%')
print(f'Core average epithermal-to-thermal ratio: {epi_to_thermal_ratio:.2f}')


