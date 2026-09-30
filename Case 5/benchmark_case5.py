import openmc as mc
import numpy as np
import matplotlib.pyplot as plt

temp = 300

# Materials
#========================================================================
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

reflector = mc.Material(name='Reflector')
reflector.add_element('C', 9.0248e-02, 'ao')
reflector.set_density('atom/b-cm', 9.0248e-02)
reflector.add_s_alpha_beta('c_Graphite')

materials = mc.Materials([fuel, shell, reflector])

# Geomitry
#========================================================================
r_fuel = 2.5
r_pebble = 3.0
box = 100      # cm
# Automatic n
pf_target = 0.61  # packing fraction (not %)
v_pebble = 4/3 * np.pi * r_pebble**3
n = round((pf_target * box**3 / (2 * v_pebble)) ** (1/3))
pitch = box / n

#BCC unit cells
h = pitch / 2
center = [(0.0, 0.0, 0.0)]
corners = [(xi * h, yi * h, zi * h) 
           for xi in (-1, 1) for yi in (-1, 1) for zi in (-1, 1)]
all_pos = center + corners

# Shapes
fuel_sph = [mc.Sphere(x0=x, y0=y, z0=z, r=r_fuel)
            for x, y, z in all_pos]
pebble_sph = [mc.Sphere(x0=x, y0=y, z0=z, r=r_pebble)
            for x, y, z in all_pos]

# Regions
fuel_region = -fuel_sph[0]
for i in fuel_sph[1:]:
    fuel_region = fuel_region | -i

pebbel_region = -pebble_sph[0] & +fuel_sph[0]
for i, j in zip(pebble_sph[1:], fuel_sph[1:]):
    pebbel_region = pebbel_region | (-i & +j)

void_region = +pebble_sph[0]
for i in pebble_sph[1:]:
    void_region = void_region & +i

# Cells
fuel_cell = mc.Cell(name='Fuel zone', region=fuel_region, fill=fuel)
shell_cell = mc.Cell(name='Graphite shell', region=pebbel_region, fill=shell)
void_cell = mc.Cell(name='Void', region=void_region)
fuel_cell.temperature = temp
shell_cell.temperature = temp

bcc_universe = mc.Universe(cells=[fuel_cell, shell_cell, void_cell])

# Lattic of BCC cells
lower_left = (-box/2,-box/2,-box/2)
lattice = mc.RectLattice(name='BCC Pebble Lattice')
lattice.lower_left = lower_left
lattice.pitch = (pitch, pitch, pitch)
lattice.universes = [[[bcc_universe for _ in range(n)]
                      for _ in range(n)]
                      for _ in range(n)]

# Surface
box_surf = mc.model.RectangularParallelepiped(
    -box/2, box/2, -box/2, box/2, -box/2, box/2
)
reflec_surf = mc.model.RectangularParallelepiped(
    -box/2 * 3, box/2 * 3, -box/2 * 3, box/2 * 3, -box/2 * 3, box/2 * 3, 
    boundary_type='vacuum'
)

# Cells
box_cell = mc.Cell(name='box', region=-box_surf, fill=lattice)
reflect_cell = mc.Cell(name='Reflector', region=(+box_surf & -reflec_surf), fill=reflector)
reflect_cell.temperature = temp

geometry = mc.Geometry([box_cell, reflect_cell])

# Tallies
#==============================================================================
epithermal = mc.EnergyFilter([1.86, 2.0e7])
thermal = mc.EnergyoutFilter([0.0, 1.86])
core_filter = mc.CellFilter([box_cell])
refl_filter = mc.CellFilter([reflect_cell])
energy_filter = mc.EnergyFilter([0.0, 1.86, 2.0e7])


# In-core scattering telly
epi_to_thermal_tally = mc.Tally(name='Epithermal to thermal scattering')
epi_to_thermal_tally.filters = [core_filter, epithermal, thermal]
epi_to_thermal_tally.scores = ['scatter']

# Leakage core to reflector tally
# Mesh filter
mesh = mc.RegularMesh()
mesh.dimension = (1, 1, 1)
mesh.lower_left = (-box/2, -box/2, -box/2)
mesh.upper_right = (box/2, box/2, box/2)
# Current (neutrons leaking from the core to the reflector)
current_tally = mc.Tally(name='Core reflector current')
current_tally.filters = [mc.MeshSurfaceFilter(mesh), energy_filter]
current_tally.scores = ['current']

# Core average spectrum
flux_core = mc.Tally(name='Core flux spectrum')
flux_core.filters = [core_filter, energy_filter]
flux_core.scores = ['flux']

# Reflector average spectrum
flux_refl = mc.Tally(name='Reflector flux spectrum')
flux_refl.filters = [refl_filter, energy_filter]
flux_refl.scores = ['flux']

# Core absorption, (core destruction rate)
core_abs = mc.Tally(name='Core absorption')
core_abs.filters = [core_filter]
core_abs.scores = ['absorption']

# Absorption in the system, for leakage out of the system
absorption = mc.Tally(name='Absorption')
absorption.scores = ['absorption']

# Flux profiles
line_mesh = mc.RegularMesh()
line_mesh.dimension = (60, 1, 1)
line_mesh.lower_left = (0, -50, -50)
line_mesh.upper_right = (150, 50, 50)

profile = mc.Tally(name='Flux profile')
profile.filters = [mc.MeshFilter(line_mesh), energy_filter]
profile.scores = ['flux']

# Relative power profile
power_mesh = mc.RegularMesh()
power_mesh.dimension = (20, 1, 1)
power_mesh.lower_left = (0, -50, -50)
power_mesh.upper_right = (50, 50, 50)

power = mc.Tally(name='Power profile')
power.filters = [mc.MeshFilter(power_mesh)]
power.scores = ['fission']


tallies = mc.Tallies([
    epi_to_thermal_tally, core_abs, current_tally, flux_core, flux_refl, absorption,   # A
    profile, power,                                                                     # B
])

# Run settings
# ===========================================================================
settings = mc.Settings()
settings.batches = 200
settings.inactive = 30
settings.particles = 10000
settings.temperature = {'method': 'interpolation'}

bounds = [-50, 50, -50, 50, -50, 50]
uniform_dist = mc.stats.Box(bounds[0::2], bounds[1::2])
settings.source = mc.IndependentSource(space=uniform_dist, 
                                       constraints={'fissionable': True})

# Export
#============================================================================
materials.export_to_xml()
geometry.export_to_xml()
settings.export_to_xml()
tallies.export_to_xml()

# Run
#============================================================================
mc.run()


# Print tallies
#============================================================================
sp = mc.StatePoint(f'statepoint.{settings.batches}.h5')

def tally_vals(name):
    return sp.get_tally(name=name).mean.flatten()

# Net current out of the core (out - in over all 6 faces)
def net_leak(df):
    surf_col = df[(f'mesh {mesh.id}', 'surf')]
    out = df[surf_col.str.endswith(' out')]['mean'].sum()
    inn = df[surf_col.str.endswith(' in')]['mean'].sum()
    return out - inn

# Bin width, cross section area and bin centres of a line mesh
def mesh_bins(m):
    dx = (m.upper_right[0] - m.lower_left[0]) / m.dimension[0]
    area = (m.upper_right[1] - m.lower_left[1]) * (m.upper_right[2] - m.lower_left[2])
    x = m.lower_left[0] + dx * (np.arange(m.dimension[0]) + 0.5)
    return dx, area, x

# Single parameters
#============================================================================
keff = sp.keff
scatter_epi_th = tally_vals('Epithermal to thermal scattering')[0]
core_abs_rate = tally_vals('Core absorption')[0]
sys_abs_rate = tally_vals('Absorption')[0]

# Leakage out of the system
gt = sp.global_tallies
leakage_rate = gt['mean'][gt['name'] == b'leakage'][0]

# Leakage core to reflector
cur_df = sp.get_tally(name='Core reflector current').get_pandas_dataframe()
net_th = net_leak(cur_df[cur_df['energy low [eV]'] == 0.0])
net_epi = net_leak(cur_df[cur_df['energy low [eV]'] == 1.86])
net_tot = net_th + net_epi

# Destruction rates (absorption + leakage)
core_destruction = core_abs_rate + net_tot
sys_destruction = sys_abs_rate + leakage_rate

# Spectra, [thermal, epithermal]
core_flux = tally_vals('Core flux spectrum')
refl_flux = tally_vals('Reflector flux spectrum')

print('=' * 64)
print('SINGLE PARAMETERS (Case 5)')
print('=' * 64)
print(f'k-effective:                                {keff.n:.5f} +/- {keff.s:.5f}')
print(f'In-core scattering epithermal -> thermal:   {scatter_epi_th / core_destruction * 100:.1f} %')
print('Leakage core -> reflector:')
print(f'   Epithermal (> 1.86 eV):                  {net_epi / core_destruction * 100:.1f} %')
print(f'   Thermal (< 1.86 eV):                     {net_th / core_destruction * 100:.1f} %')
print(f'   Total:                                   {net_tot / core_destruction * 100:.1f} %')
print(f'Total leakage out of the system:            {leakage_rate / sys_destruction:.4f}')
print(f'Core average epithermal/thermal ratio:      {core_flux[1] / core_flux[0]:.3f}')
print(f'Reflector average epithermal/thermal ratio: {refl_flux[1] / refl_flux[0]:.4f}')

# Line profiles
#============================================================================
# Flux profiles, per cm^3
dx_l, area_l, x_line = mesh_bins(line_mesh)
prof = sp.get_tally(name='Flux profile').mean.reshape(line_mesh.dimension[0], 2)
vol_l = dx_l * area_l
thermal_profile = prof[:, 0] / vol_l
epithermal_profile = prof[:, 1] / vol_l
ratio_profile = np.divide(epithermal_profile, thermal_profile,
                          out=np.zeros_like(thermal_profile),
                          where=thermal_profile > 0)

# Relative power profile (average = 1.0)
dx_p, area_p, x_pow = mesh_bins(power_mesh)
power_raw = tally_vals('Power profile')
relative_power = power_raw / power_raw.mean()

# Save and plot
#============================================================================
np.savetxt('case5_flux_profiles.csv',
           np.column_stack([x_line, epithermal_profile, thermal_profile, ratio_profile]),
           delimiter=',', header='x_cm,epithermal,thermal,ratio', comments='')
np.savetxt('case5_power_profile.csv',
           np.column_stack([x_pow, relative_power]),
           delimiter=',', header='x_cm,relative_power', comments='')

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].plot(x_line, epithermal_profile)
ax[0, 0].set_title('Epithermal flux (> 1.86 eV)')
ax[0, 0].set_ylabel('Flux (n/cm$^2$ per source n)')
ax[0, 1].plot(x_line, thermal_profile)
ax[0, 1].set_title('Thermal flux (< 1.86 eV)')
ax[0, 1].set_ylabel('Flux (n/cm$^2$ per source n)')
ax[1, 0].plot(x_line, ratio_profile)
ax[1, 0].set_title('Epithermal / thermal ratio')
ax[1, 0].set_ylabel('Epithermal / thermal (-)')
ax[1, 1].plot(x_pow, relative_power)
ax[1, 1].set_title('Relative power (core)')
ax[1, 1].set_ylabel('Relative power (-)')

# Core-reflector interface
for a in ax.flat[:3]:
    a.axvline(box / 2, color='grey', ls='--')
for a in ax.flat:
    a.set_xlabel('Distance from centre (cm)')

plt.tight_layout()
plt.show()
