import numpy as np
import openmc as mc
import matplotlib.pyplot as plt

temp = 300

# Material
# =========================

fuel = mc.Material(name='Pebble fuel zone (homogenised)')
fuel.add_nuclide('U234', 3.3090e-07, 'ao')
fuel.add_nuclide('U235', 3.3812e-05, 'ao')
fuel.add_nuclide('U238', 3.1437e-04, 'ao')
fuel.add_element('Si', 7.8039e-04, 'ao')
fuel.add_element('C', 8.4743e-02, 'ao')
fuel.add_nuclide('O16', 6.9636e-04, 'ao')
fuel.add_element('B', 2.4795e-08, 'ao')
fuel.set_density('atom/b-cm', 3.3090e-07 + 3.3812e-05 + 3.1437e-04
                 + 7.8039e-04 + 8.4743e-02 + 6.9636e-04 + 2.4795e-08)
fuel.add_s_alpha_beta('c_Graphite')
 
shell = mc.Material(name='Pebble graphite shell')
shell.add_element('C', 8.7240e-02, 'ao')
shell.add_element('B', 2.4795e-08, 'ao')
shell.set_density('atom/b-cm', 8.7240e-02 + 2.4795e-08)
shell.add_s_alpha_beta('c_Graphite')
 
materials = mc.Materials([fuel, shell])

# Geometry
# =========================
# Dimensions
r_Fule = 2.5   # fule radius
r_pebble = 3.0 # pebble radius
n = 14
box = 100 # cm
pitch = box / n

# BCC unit cell: one pebble at the centre + 8 corner pebbles.
# The lattice cuts the corner pebbles, so each cell holds 8 octants and
# neighbouring cells together rebuild the full corner pebbles.
# Nearest-neighbour distance = pitch*sqrt(3)/2 = 6.19 cm > 6.0 cm (no overlap).
h = pitch / 2
centres = [(0.0, 0.0, 0.0)]
corners = [(sx * h, sy * h, sz * h)
           for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
all_pos = centres + corners

fuel_spheres = [mc.Sphere(x0=x, y0=y, z0=z, r=r_Fule) for x, y, z in all_pos]
peb_spheres = [mc.Sphere(x0=x, y0=y, z0=z, r=r_pebble) for x, y, z in all_pos]

# Fuel zone: union of the 9 fuel spheres
fuel_region = -fuel_spheres[0]
for s in fuel_spheres[1:]:
    fuel_region = fuel_region | -s

# Shell: union of (inside pebble sphere, outside fuel sphere)
shell_region = (-peb_spheres[0] & +fuel_spheres[0])
for sp, sf in zip(peb_spheres[1:], fuel_spheres[1:]):
    shell_region = shell_region | (-sp & +sf)

# Void between pebbles: outside every pebble sphere
void_region = +peb_spheres[0]
for s in peb_spheres[1:]:
    void_region = void_region & +s

fuel_cell = mc.Cell(name='fuel zone', region=fuel_region, fill=fuel)
shell_cell = mc.Cell(name='graphite shell', region=shell_region, fill=shell)
void_cell = mc.Cell(name='void between pebbles', region=void_region)  # void
fuel_cell.temperature = temp
shell_cell.temperature = temp

bcc_universe = mc.Universe(cells=[fuel_cell, shell_cell, void_cell])

# Lattice of BCC cells filling the box
# =========================================================================
lattice = mc.RectLattice(name='BCC pebble lattice')
lattice.lower_left = (-box / 2, -box / 2, -box / 2)
lattice.pitch = (pitch, pitch, pitch)
lattice.universes = [[[bcc_universe for _ in range(n)]
                      for _ in range(n)] for _ in range(n)]

box_surface = mc.model.RectangularParallelepiped(
    -box / 2, box / 2, -box / 2, box / 2, -box / 2, box / 2,
    boundary_type='reflective')  # try 'white' as well (report wording)
box_cell = mc.Cell(name='box', region=-box_surface, fill=lattice)

geometry = mc.Geometry([box_cell])

# Packing fraction check (report specifies 61 %)
# =========================================================================
v_peb = 4 / 3 * np.pi * r_pebble ** 3
packing = 2 * n ** 3 * v_peb / box ** 3
print(f'Pebbles per cell: 2, pitch = {pitch:.5f} cm, packing fraction = {packing:.4f}')


# Plot
# =======================================
geometry.plot(
    basis='xy',
    origin=(0, 0, 0),
    width=(box, box),
    pixels=(1600, 1600),
    color_by='cell',
    colors={
        fuel_cell: (200, 60, 60),
        shell_cell: (120, 120, 120),
        void_cell: (255, 255, 255),
    },
)
plt.show()

# In-core scattering epithermal -> thermal
# =======================================
energy_in_epithermal = mc.EnergyFilter([1.86, 2.0e7])
energy_out_thermal = mc.EnergyoutFilter([0.0, 1.86])

epi_to_thermal_tally = mc.Tally(name='Epithermal to thermal scattering')
epi_to_thermal_tally.filters = [energy_in_epithermal, energy_out_thermal]
epi_to_thermal_tally.scores = ['scatter']

absorption_tally = mc.Tally(name='Absorption')
absorption_tally.scores = ['absorption']

# Core average spectra
flux_energy_filter = mc.EnergyFilter([0.0, 1.86, 2.0e7])
flux_tally = mc.Tally(name='Flux Spectrum')
flux_tally.filters = [flux_energy_filter]
flux_tally.scores = ['flux']

tallies = mc.Tallies([epi_to_thermal_tally, absorption_tally, flux_tally])

# Settings
# =======================================
settings = mc.Settings()
settings.batches = 100
settings.inactive = 30
settings.particles = 5000
settings.temperature = {'method': 'interpolation'}

bounds = [-5, 5, -5, 5, -5, 5]
uniform_dist = mc.stats.Box(bounds[:3], bounds[3:])
settings.source = mc.IndependentSource(space=uniform_dist,)

# Export
# =======================================
materials.export_to_xml()
geometry.export_to_xml()
settings.export_to_xml()
tallies.export_to_xml()

# Run
# =======================================
mc.run()

# Print Tallies
# =======================================
sp = mc.StatePoint(f'statpoint.{settings.batches}.h5')

epi_to_thermal_scatter = sp.get_tally(name='Epithermal to thermal scattering')
absorption_rate = sp.get_tally(name='Absorption').mean.flatten()[0]
destruction_rate = absorption_rate

flux_df = sp.get_tally(name='Flux Spectrum').get_pandas_dataframe()
thermal_flux = flux_df['mean'][0]
epithermal_flux = flux_df['mean'][1]

pct_epi_thermal = epi_to_thermal_scatter / destruction_rate * 100
epi_to_thermal_ratio = epithermal_flux / thermal_flux

print(f'In-core scattering epithermal -> thermal: {pct_epi_thermal:.1f}%')
print(f'Core average epithermal-to-thermal ratio: {epi_to_thermal_ratio:.2f}')