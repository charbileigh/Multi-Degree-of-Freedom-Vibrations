# Name: Francesca Seopa
# MEC4047F - Project 2: MDOF 
# Project 2 Part 2

# Reference: Part of this code was referenced from Andrew Friedman - a Mechanical Engineer


'''
    The purpose of this project is to develop a numerical Solution in Python that can solve
    the displacement-time response for a general N-DOF system. This will be used to find the
    displacement and velocity histories of the Input Motor and Output Rotor as well as any other
    relevant DOF (of any system that is required to be solved). 

'''


# Importing libraries for the entire programme
import numpy as np
from math import sin
from scipy.linalg import eigh
from numpy.linalg import inv
from matplotlib import pyplot as plt



# Setting up parameters for an example other than the question
# given in the project to prove the validity of the code
F0 = 5.0
omega = 48.599999
k = 1000.0
m = 4.0
dof = 3                                 # The DOF can be manually set



# These time steps are measured in seconds [s]
time_step = 1.0e-4                      # small time step for more accuracy
start_time = 0
end_time = 5.0                          # time step that can be manually altered


# setting up matrices to form the K,M,I matrices.
# A C matrix can be set up for other problems. However,
# for this specific example the C matrix was not used, hence the C
# matrix being commented out
K_matrix = np.array([[3*k,-k,-k],[-k,k,0],[-k,0,k]])
M_matrix = np.array([[2*m,0,0],[0,m,0],[0,0,m]])
I_matrix = np.identity(dof)
# C_matrix = np.array([[c,0,0],[0,c,0],[0,0,c]])



# initializing matrices according to the degree of freedom chosen
A = np.zeros((2*dof,2*dof))
B = np.zeros((2*dof,2*dof))
Y = np.zeros((2*dof,1))
F = np.zeros((2*dof,1))


# setting up constraints for the M and I matrices for calculations
A[0:3,0:3] = M_matrix
A[3:6,3:6] = I_matrix


# constraining the system matrices for further calculations
B[0:3,3:6] = K_matrix
B[3:6,0:3] = -I_matrix


# initializing the matrices to be used in calculating the EOM 
A_inv = inv(A)
force = []
X1_matrix = []
X2_matrix = []
X3_matrix = []



# finding the natural frequencies and mode shapes
evals, evecs = eigh(K_matrix,M_matrix)   # determining the eigenvalues and eigenvectors     
frequencies = np.sqrt(evals)             # getting the frequencies from the sqrt of the eigenvectors
print("Frequencies of the problem set given above indicating the modal shapes:\n",
      frequencies)
print()
print()
print("Eigen values of the problem set given above:\n", evecs)




# numerically integrating the EOMs from the matrices defined
# and the initialized empty matrices and force matrix
for t in np.arange(0, end_time, time_step):
	F[1] = F0 * sin(omega*t)
	Y_new = Y + time_step * A_inv.dot( F - B.dot(Y) )
	Y = Y_new
	force.extend(F[1])
	X1_matrix.extend(Y[3])
	X2_matrix.extend(Y[4])
	X3_matrix.extend(Y[5])


# plotting the results calculated above
time = [round(t,5) for t in np.arange(0, end_time, time_step) ]


# plotting the time graphs of the various masses/forces
plt.plot(time,X1_matrix)
plt.plot(time,X2_matrix)
plt.plot(time,X3_matrix)


# labelling the plots for clarity
plt.xlabel('time (s)')
plt.ylabel('displacement (m)')
plt.title('Response Curves')
plt.legend(['X1_matrixPoints', 'X2_matrixPoints', 'X3_matrixPoints'], loc='lower right')
plt.show()


# The second time plot of the problem set
plt.plot(time,force)
plt.show()

