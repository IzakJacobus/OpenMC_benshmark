import openmc as mc
import numpy as np


temp = 1200

# Materials
#======================================================

kernal = mc.Material(name='UO2 Kernal')
kernal.add_nuclide('U234', 2.2057e-05, 'ao')
kernal.add_nuclide('U235', 2.2539e-03,'ao')
kernal.add_nuclide('U238', 2.0956e-02, 'ao')
kernal.add_nuclide('O16', 4.6421e-02, 'ao')
kernal.add_element('B', 2.4795e-08, 'ao')
kernal.set_density('atom/b-cm', 2.2057e-05 + 2.2539e-03 
               + 2.0956e-02 + 4.6421e-02 + 2.4795e-08)

coating1 = mc.Material(name='Buffer 1')
coating1.add_element('C', 5.2645e-02, 'ao')
coating1.add_element('B', 2.4795e-08, 'ao')
coating1.set_density('atom/b-cm', 5.2645e-02 + 2.4795e-08)
coating1.add_s_alpha_beta('c_Graphite')

coating2 = mc.Material(name='Inner PyC')
coating2.add_element('C', 9.5262e-02, 'ao')
coating2.add_element('B', 2.4795e-08, 'ao')
coating2.set_density('atom/b-cm', 9.5262e-02 + 2.4795e-08)
coating2.add_s_alpha_beta('c_Graphite')

coating3 = mc.Material(name='SiC')
coating3.add_element('C', 4.7760e-02, 'ao')
coating3.add_element('Si', 4.7760e-02, 'ao')
coating3.add_element('B', 2.4795e-08, 'ao')
coating3.set_density('atom/b-cm', 2 * 4.7760e-02 + 2.4795e-08)

coating4 = mc.Material(name='Outer PyC')
coating4.add_element('C', 9.5262e-02, 'ao')
coating4.add_element('B', 2.4795e-08, 'ao')
coating4.set_density('atom/b-cm', 9.5262e-02 + 2.4795e-08)
coating4.add_s_alpha_beta('c_Graphite')

matrix = mc.Material(name='Matrix filling')
matrix.add_element('C', 8.7240e-2, 'ao')
matrix.add_element('B', 2.4795e-8, 'ao')
matrix.set_density('atom/b-cm', 8.7240e-2 + 2.4795e-08)
matrix.add_s_alpha_beta('c_Graphite')
matrix.temperature = temp


materials = mc.Materials([kernal,coating1,coating2,coating3,
                          coating4,matrix])

# Geomitry
#======================================================================

# Dimentions
r_fuel = 2.5
r_pebble = 3.0
r_kernal = 0.0250
r_buffer = 0.0345
r_IPyC = 0.0385
r_SiC = 0.0420
r_OPyC = 0.046
box = 100       # cm
# Take the target pakking fraction and determane the number of Bcc unit cells
# allong each side (n).
pf_target = 0.61  # packing fraction (not %)
v_pebble = 4/3 * np.pi * r_pebble**3
n = round((pf_target * box**3 / (2 * v_pebble)) ** (1/3))
pitch = box / n

# Particle
s_kernal = mc.Sphere(r=r_kernal)
s_buffer = mc.Sphere(r=r_buffer)
s_IPyC = mc.Sphere(r=r_IPyC)
s_SiC = mc.Sphere(r=r_SiC)
s_OPyC = mc.Sphere(r=r_OPyC)


# Particle Composition
kernal_cell = mc.Cell(name='Kernal Zone', region=-s_kernal, fill=kernal)
buffer_cell = mc.Cell(name='Buffer Zone', region=(-s_buffer & +s_kernal),
                       fill=coating1)
IPyC_cell = mc.Cell(name='Inner PyC Zone', region=(-s_IPyC & + s_buffer),
                     fill=coating2)
SiC_cell = mc.Cell(name='SiC Zone', region=(-s_SiC & +s_IPyC),
                    fill=coating3)
OPyC_cell = mc.Cell(name='Outter PyC Zone', region=(-s_OPyC & +s_SiC),
                     fill=coating4)

for i in (kernal_cell, buffer_cell, IPyC_cell, SiC_cell, OPyC_cell):
    i.temperature = temp

particle_universe = mc.Universe(name='One particle universe',
                                cells=[kernal_cell, buffer_cell,
                                        IPyC_cell, SiC_cell, OPyC_cell])


# Safty chack
nearest_neighbour_dist = pitch * np.sqrt(3) / 2
assert nearest_neighbour_dist > 2 * r_pebble, (
    f'Pebbles overlap: nearest-neighbour distance {nearest_neighbour_dist:.3f} cm '
    f'<= pebble diameter {2 * r_pebble:.3f} cm. Reduce n_cells_per_side.'
)

# Create a cell with the 9 spheres
h = pitch / 2
center = [(0.0, 0.0, 0.0)]
corners = [(sx * h, sy * h, sz* h)
           for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
all_pos = center + corners

# Create the pebbles
fuel_sph = [mc.Sphere(x0=x, y0=y,z0=z, r=r_fuel)
            for x, y, z in all_pos]
pebble_sph = [mc.Sphere(x0=x, y0=y,z0=z, r=r_pebble)
              for x, y, z in all_pos]


# Pebble regions
fuel_region = -fuel_sph[0]
for s in fuel_sph[1:]:
    fuel_region = fuel_region | -s

pebble_region = (-pebble_sph[0] & +fuel_sph[0])
for s1, s2 in zip(pebble_sph[1:], fuel_sph[1:]):
    pebble_region = pebble_region | (-s1 & +s2)

void_region = +pebble_sph[0]
for s in pebble_sph[1:]:
    void_region = void_region & +s

# Create random positions of the particles to occupy
# pack_spheres needs a single simple shape, so pack one pebble at the origin
# and reuse it as a universe at every BCC position.
s_fuel_origin = mc.Sphere(r=r_fuel)
centers = mc.model.pack_spheres(radius=r_OPyC,
                                region=-s_fuel_origin,
                                pf=0.093)

particles = [mc.model.TRISO(r_OPyC, particle_universe, c) 
             for c in centers]

# Makking a latice
lower_left = (-r_fuel, -r_fuel, -r_fuel)
shape = (10, 10, 10)
lattice_pitch = 2 * r_fuel / 10

fuel_lattice = mc.model.create_triso_lattice(
    particles,
    lower_left,
    (lattice_pitch,lattice_pitch,lattice_pitch),
    shape,
    matrix
)

# One pebble (centred at origin)
s_pebble_origin = mc.Sphere(r=r_pebble)
pebble_fuel_cell = mc.Cell(name='fuel zone', region=-s_fuel_origin,
                           fill=fuel_lattice)
pebble_shell_cell = mc.Cell(name='graphite shell',
                            region=-s_pebble_origin & +s_fuel_origin,
                            fill=matrix)
pebble_fuel_cell.temperature = temp
pebble_universe = mc.Universe(name='Pebble',
                              cells=[pebble_fuel_cell, pebble_shell_cell])

# Place the pebble universe at each of the 9 BCC positions
pebble_cells = []
for i, (sph, pos) in enumerate(zip(pebble_sph, all_pos)):
    cell = mc.Cell(name=f'pebble {i}', region=-sph, fill=pebble_universe)
    cell.translation = pos
    pebble_cells.append(cell)
void_cell = mc.Cell(name='void between pebbles', region=void_region)  # void

bcc_universe = mc.Universe(cells=pebble_cells + [void_cell])

# Filling the box with the  bcc_ universe
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
settings.source = mc.IndependentSource(space=uniform_dist, constraints={'fissionable': True})
settings.source_rejection_fraction = 0.001  # kernels are only ~0.5% of the volume

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
sp = mc.StatePoint(f'statepoint.{settings.batches}.h5')

epi_to_thermal_scatter = sp.get_tally(name='Epithermal to thermal scattering').mean.flatten()[0]
absorption_rate = sp.get_tally(name='Absorption').mean.flatten()[0]
destruction_rate = absorption_rate

flux_df = sp.get_tally(name='Flux Spectrum').get_pandas_dataframe()
thermal_flux = flux_df['mean'][0]
epithermal_flux = flux_df['mean'][1]

pct_epi_thermal = epi_to_thermal_scatter / destruction_rate * 100
epi_to_thermal_ratio = epithermal_flux / thermal_flux

print(f'In-core scattering epithermal -> thermal: {pct_epi_thermal:.1f}%')
print(f'Core average epithermal-to-thermal ratio: {epi_to_thermal_ratio:.2f}')