#!/usr/bin/env python

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import copy


class classical_particles():
    def __init__(self):
        self.atom_colors = {'Mg': 'salmon','O': 'red', 'V':'blue'}		# Set up colors of atoms
        self.atom_radii  = {'Mg': 35,'O': 55, 'V': 65}				# Set up atomic radii

        self.V0 = -1.2								# Charges of vacancy: -1.2 for neutral, -0.2 for V+ and 0.8 for +2
        self.V1 = -1.2                                
        O = -1.2								# O partial charge
        Mg = 1.2								# Mg partial charge
        
        MgO = 2.1								# MgO (and MgV) distance
        self.r = (2.1**2 *2)**0.5						# OO (and VO) distance
        
        self.particles = [ [O,  [self.r, self.r/2.], 1,'O'],                    # Set up initial particles. [0] = charge, [1] = positions, [2] = weight, [3] = name
                           [Mg, [self.r/2., self.r], 1,'Mg'],
                           [Mg, [self.r/2., 0], 1,'Mg'],
                           [self.V0, [self.r, self.r/2.], 0.0,'V'],
                           [self.V1, [0, self.r/2.], 1.0,  'V'] ]
        
        
        return None

    def particles2(self):
        '''
        Set up second system of particles where V has migrated
        '''
        self.particles_2 = copy.deepcopy(self.particles)
        self.particles_2[0][1][0] = 0.0                                      	# Move Vacancy
        return None
        
    def plot_particles(self):        
        '''
        Plot the positions of the particles in self.particles. Their size is dictated by their atomic radii
        '''

        self.fig_particles, self.ax_particles = plt.subplots(figsize=(15,10))        	# Set up Matplotlib figure
        for particle in self.particles:							# Plot particles
            self.ax_particles.plot(particle[1][0],particle[1][1],'o',ms=self.atom_radii[particle[3]]*particle[2],color = self.atom_colors[particle[3]], markeredgecolor='k')
        particle_positions = np.array([i[1] for i in self.particles])

        # Adjust axes limits
        minlimits = particle_positions.min(axis=0)
        maxlimits = particle_positions.max(axis=0)
        self.ax_particles.set_xlim((minlimits[0]-0.2,maxlimits[0]+0.2))
        self.ax_particles.set_ylim((minlimits[1]-0.2,maxlimits[1]+0.2))
        self.ax_particles.set_xlabel('x position')
        self.ax_particles.set_ylabel('y position')
        return None

    def calculate_dipole_migration(self):
        '''
	Move atoms from particles to particles2. Direction must be defined (usually set to direction of motion). A new variable containing the dipole moments along the migration path is created. Particles2 must exist otherwise it will fail.
	'''
        direction = np.array([1,0])
        positions1 = np.array([i[1] for i in self.particles])
        positions2 = np.array([i[1] for i in self.particles_2])
        self.dipole_moments = []
        self.dipole_magnitudes = []
        nInterp = 11
        displacements = (positions2 - positions1)/(nInterp - 1)
        for i in xrange(nInterp):

            self.particles[4][2] = (erf(((1-float(i)/(nInterp-1))-0.5)*5) / 2.0)+0.5 
            self.particles[3][2] = (erf(((float(i)/(nInterp-1))-0.5)*5) / 2.0) + 0.5 
            dipole_moment = [] 
            for ind,particle in enumerate(self.particles):
                dipole_moment.append(particle[0] * particle[2] * np.array(particle[1]))
            dipole_moment = np.array(dipole_moment)
            self.dipole_moments.append(dipole_moment)
            angle = np.dot(np.sum(dipole_moment,axis=0),direction) / (np.linalg.norm(direction) * np.linalg.norm(np.sum(dipole_moment,axis=0)))
            if angle < 0:
                self.dipole_magnitudes.append(-np.linalg.norm(dipole_moment.sum(axis=0))) 
            else:
                self.dipole_magnitudes.append(np.linalg.norm(dipole_moment.sum(axis=0))) 
            for ind,particle in enumerate(self.particles):
                particle[1] = particle[1] +  displacements[ind]
        self.dipole_moments = np.array(self.dipole_moments)
        return None 

    def plot_migration_dipoles(self):
        '''
	Plot the dipoles along the migration trajectory. Dipole moments must have been calculated otherwise it will fail.
        '''
        x = 0
        nInterp = 11
        self.fig_dipole, self.ax_dipole = plt.subplots()
        for frame in xrange(nInterp):
            for particle in xrange(len(self.dipole_moments[frame])): 
                self.ax_dipole.quiver(x,0,self.dipole_moments[frame][particle][0],self.dipole_moments[frame][particle][1],angles='xy',scale_units='xy',scale=1,color = self.atom_colors[self.particles[particle][3]])
            x += 5
        self.ax_dipole.set_ylim((self.dipole_moments.min(axis=0)[:,1].min(), self.dipole_moments.max(axis=0)[:,1].max()))
        self.ax_dipole.set_xlim((-5,nInterp*5+5))

        self.ax_dipole.set_xlabel('Migration trajectory')
        self.ax_dipole.set_ylabel('Dipole Moment')


        self.fig_dipole_magnitude, self.ax_dipole_magnitude = plt.subplots()
        self.ax_dipole_magnitude.plot(self.dipole_magnitudes)
        self.ax_dipole_magnitude.set_xlabel('Migration trajectory')
        self.ax_dipole_magnitude.set_ylabel('Dipole Moment')
        return None


if __name__ == '__main__':
    system = classical_particles()           	# Set up classical system
    system.particles2()				# Calculate particles2
    system.plot_particles()			# Plot the system of particles
    system.calculate_dipole_migration()         # Calculate dipole along migration trajectory
    system.plot_migration_dipoles()             # Plot the dipole migration along the trajectory

    # Save figures
    if True:
        system.fig_particles.savefig('particles.png')
        system.fig_dipole.savefig('dipole_migration.png')
        system.fig_dipole_magnitude.savefig('dipole_magnitudes.png')
 
