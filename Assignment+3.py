
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets
import scipy.integrate as sint
import sympy


# In[2]:


HI, HII, HeI, HeII, HeIII, de, Hm, H2I, H2II = sympy.sympify(
"HI, HII, HeI, HeII, HeIII, de, Hm, H2I, H2II")
k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31 = sympy.sympify(
"k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31")


# In[3]:


# The follow reaction rates were found here:
# http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1991ApJS...76..759L&data_type=PDF_HIGH&whole_paper=YES&type=PRINTER&filetype=.pdf
# https://arxiv.org/pdf/astro-ph/0003212.pdf

# H + e -> HI + 2e
def k1(T):
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)

    rv = np.exp(-32.71396786375
          + 13.53655609057*log_T_eV
          - 5.739328757388*log_T_eV**2 
          + 1.563154982022*log_T_eV**3
          - 0.2877056004391*log_T_eV**4
          + 0.03482559773736999*log_T_eV**5
          - 0.00263197617559*log_T_eV**6
          + 0.0001119543953861*log_T_eV**7
          - 2.039149852002e-6*log_T_eV**8)
    return rv

# HI + e -> H + gamma
def k2(T):
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)
    rv = np.exp(-28.6130338-0.72411256*log_T_eV
                -2.02604473e-2*log_T_eV**2
                -2.38086188e-3*log_T_eV**3
                -3.21260521e-4*log_T_eV**4
                -1.42150291e-5*log_T_eV**5
                +4.98910892e-6*log_T_eV**6
                +5.75561414e-7*log_T_eV**7
                -1.8567670e-8*log_T_eV**8
                -3.07113524e-9*log_T_eV**9)
    return rv
    
# He + e ->  HeI + 2e
def k3(T):
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)
    rv = np.exp(-44.09864886 
          + 23.91596563*log_T_eV 
          - 10.7532302*log_T_eV**2 
          + 3.05803875*log_T_eV**3 
          - 0.56851189*log_T_eV**4
          + 6.79539123e-2*log_T_eV**5
          - 5.00905610e-3*log_T_eV**6 
          + 2.06723616e-4*log_T_eV**7
          - 3.64916141e-6*log_T_eV**8)
    return rv 

# HeI + e -> He + gamma
def k4(T):
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)
    rv = 3.925e-13*T_eV**(-0.6353) 
    return rv 
          
# HeI + e -> HeII + 2e
def k5(T):
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)
    rv = np.exp(-68.71040990 
             + 43.93347633*log_T_eV
             - 18.4806699*log_T_eV**2 
             + 4.70162649*log_T_eV**3
             - 0.76924663*log_T_eV**4 
             + 8.113042e-2*log_T_eV**5
             - 5.32402063e-3*log_T_eV**6 
             + 1.97570531e-4*log_T_eV**7
             - 3.16558106e-6*log_T_eV**8)
    return rv 

# HeII + e -> HeI + gamma
# T is in K 
def k6(T):
    rv = 3.36e-10*T**(-1.2)*(T/1000)**(-0.2)*(1+(T/10**6)**0.7)**(-1)
    return rv 

# H + H -> HI + e + H
def k7(T):
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)
    rv = 1.7e-4* np.exp(-32.71396786375
                         + 13.53655609057*log_T_eV
                         - 5.739328757388*log_T_eV**2 
                         + 1.563154982022*log_T_eV**3
                         - 0.2877056004391*log_T_eV**4
                         + 0.03482559773736999*log_T_eV**5
                         - 0.00263197617559*log_T_eV**6
                         + 0.0001119543953861*log_T_eV**7
                         - 2.039149852002e-6*log_T_eV**8)
    return rv

# H + He -> HI + e + He
def k8(T):
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)
    rv = 1.75e-17*((T_eV**1.3)*np.e**(-157800/T_eV))
    return rv

# H + gamma -> HI + e
def k9(T):
    return 0

# He + gamma -> HeI + e
def k10(T):
    return 0

# HeI + gamma -> HeII + e
def k11(T):
    return 0

# H + e -> H- + gamma
def k12(T):
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)
    
    if T <= 6000:
        rv = 1.429e-18*T**0.7620*T**(0.1523*np.log10(T))*T**(-3.274e-2)*np.log10(T)**2
    else:
        rv = 3.802e-17*T**(0.1998*np.log10(T))*10**(4.0415e-5*np.log10(T)**6 - 5.447e-3*np.log10(T)**4)
    return rv     

# H- + H -> H2 + e
def k13(T):
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)
    
    if T_eV > 0.1:
        rv = np.exp(-20.06913897 
                    + 0.22898*log_T_eV 
                    + 3.5998377e-2*log_T_eV**2
                    - 4.55512e-3*log_T_eV**3
                    - 3.10511544e-4*log_T_eV**4
                    + 1.0732940*10e-4*log_T_eV**5
                    - 8.36671960*10e-6*log_T_eV**6
                    + 2.23830623*10e-7*log_T_eV**7)
    else:
        rv = 1.428e-9
    return rv

# H + HI -> H2I + gamma
def k14(T):
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)
    
    if T_eV < 0.577:
        rv = 3.833e-16*T_eV**1.8
    else:
        rv = 5.81e-16*(0.20651*T_eV)**(-0.2891*np.log(0.20651*T_eV))
    return rv 

# H2I + H -> H2 + HI
def k15(T):
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)
    rv = 6.4e-10
    return rv

# H2 + HI -> H2I + H
# How to get exponent of e? 
def k16(T):
    T_ev = T/11605
    log_T_ev = np.log(T_ev)
    rv = np.e**(-24.24914687 + 3.40082444*log_T_ev
                -3.89800396*log_T_ev**2 + 2.04558782*log_T_ev**3 
                -0.541618285*log_T_ev**4 + 8.41077503e-2*log_T_ev**5 
                -7.87902615e-3*log_T_ev**6 + 4.13839842e-4*log_T_ev**7 
                -9.36345888e-6*log_T_ev**8)
    
    return rv 
    

# H2 + e -> 2H + e
# T is in K 
def k17(T):
    rv = 5.6e-11*T**0.5*np.exp(-102124/T)
    return rv 

    
# H2 + H -> 3H
def k18(T):
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)
    rv = 1.067e-10*T_eV**2.012*np.exp(-(4.463/T_eV)*(1 + 0.2472*T_eV)**3.512)
    return rv
    
    
# H- + e -> H + 2e
def k19(T): 
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)
    rv = np.exp(-18.01849334 
             + 2.3608522*log_T_eV
             - 0.28274430*log_T_eV**2 
             + 1.62331664e-2*log_T_eV**3
             - 3.36501203e-2*log_T_eV**4
             + 1.17832978e-2*log_T_eV**5
             - 1.65619470e-3*log_T_eV**6
             + 1.06827520e-4*log_T_eV**7
             - 2.63128581e-6*log_T_eV**8)
    return rv
    
# H- + H -> 2H + e
def k20(T):
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)
    
    if T_eV > 0.1:
        rv = np.exp(-20.37260896 
                    + 1.13944933*log_T_eV
                    - 0.14210135*log_T_eV**2 
                    + 8.4644554e-3*log_T_eV**3
                    - 1.4327641e-3*log_T_eV**4
                    + 2.0122503e-4*log_T_eV**5
                    + 8.6639632e-5*log_T_eV**6
                    - 2.5850097e-5*log_T_eV**7
                    + 2.4555012e-6*log_T_eV**8
                    - 8.0683825e-8*log_T_eV**9)
    else:
        rv = 2.5634e-9*T_eV**1.78186
        
    return rv
    
    
# H- + HI -> 2H
# T is in K
def k21(T):
    rv = 7e-8*(T/100)**(-0.5)
    return rv
    
# H- + HI -> H2I + e
def k22(T):
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)
    
    if T_eV < 1.719:
        rv = 2.291e-10*T_eV**(-0.4)
    else:
        rv = 8.4258e-10*T_eV**(-1.4)*np.exp(-1.301/T_eV)
    return rv

# H2I + e -> 2H
# T is in K
def k23(T):
    if T < 617:
        rv = 1e-8
    else:
        rv = 1.32e-6*T**(-0.76)
    return rv
    
# H2I + H- -> H2 + H
# T is in K
def k24(T):
    rv = 5e-7*(100/T)**0.5
    return rv
    
# 3H -> H2 + H
def k25(T):
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)
    rv = 5.5e-29*T_eV**(-1)
    return rv

# 2H + H2 -> H2 + H2
def k26(T):
    T_eV = T / 11605.       
    log_T_eV = np.log(T_eV)
    rv = 5.5e-29*T_eV**(-1/8)
    return rv 

# H- + gamma -> H + e
def k27(T):
    return 0
    
# H2I + gamma -> H + HI
def k28(T):
    return 0
    
# H2 + gamma ->  H2I + e
def k29(T): 
    return 0    
    
# H2I + gamma -> 2HI + e
def k30(T):
    return 0
    
# H2 + gamma -> 2H
def k31(T):
    return 0    

T = int(input("Enter a temperature in Kelvin: "))
k1 = k1(T)
k2 = k2(T)
k3= k3(T)
k4= k4(T)
k5= k5(T)
k6= k6(T)
k7= k7(T)
k8= k8(T)
k9= k9(T)
k10= k10(T)
k11= k11(T)
k12= k12(T)
k13= k13(T)
k14= k14(T)
k15= k15(T)
k16= k16(T)
k17= k17(T)
k18= k18(T)
k19= k19(T)
k20= k20(T)
k21= k21(T)
k22= k22(T)
k23= k23(T)
k24= k24(T)
k25= k25(T)
k26= k26(T)
k27= k27(T)
k28= k28(T)
k29= k29(T)
k30= k30(T)
k31= k31(T)  


# In[4]:


# Reactions from Table 3
# https://arxiv.org/pdf/1610.09591.pdf
r1 = (HI + de), (HII + de + de), k1
r2 = (HII + de), (HI), k2
r3 = (HeI + de), (HeII + de + de), k3
r4 = (HeII + de), (HeI), k4
r5 = (HeII + de), (HeIII + de + de), k5
r6 = (HeIII + de), (HeII), k6
r7 = (HI + HI), (HII + de + HI), k7
r8 = (HI + HeI), (HII + de + HeI), k8
r9 = (HI), (HII + de), k9
r10 = (HeI), (HeII + de), k10
r11 = (HeII), (HeIII + de), k11

# Reactions from Table 4
# https://arxiv.org/pdf/1610.09591.pdf
r12 = (HI + de), (Hm), k12
r13 = (Hm + HI), (H2I + de), k13
r14 = (HI + HII), (H2II), k14
r15 = (H2II + HI), (H2I + HII), k15
r16 = (H2I + HII), (H2II + HI), k16
r17 = (H2I + de), (HI + HI + de), k17
r18 = (H2I + HI), (HI + HI + HI), k18
r19 = (Hm + de), (HI + de + de), k19
r20 = (Hm + HI), (HI + de + HI), k20
r21 = (Hm + HII), (HI + HI), k21
r22 = (Hm + HII), (H2II + de), k22
r23 = (H2II + de), (HI + HI), k23
r24 = (H2II + Hm), (H2I + HI), k24
r25 = (HI + HI + HI), (H2I + HI), k25
r26 = (HI + HI + H2I), (H2I + H2I), k26
r27 = (Hm), (HI + de), k27
r28 = (H2II), (HI + HII), k28
r29 = (H2I), (H2II + de), k29
r30 = (H2II), (HI + HI + de), k30
r31 = (H2I), (HI + HI), k31 

all_reactions = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31]


# In[5]:


def rhs(t, state):
    # state0 = HI, state1 = HII, state2 = HeI, state3 = HeII, state4 = HeIII
    # state5 = H2I, state6 = H2II, state7 = Hm, state8 = de, state9 = T
    
    # dnHIdt = 
    # -2*H2I*HI*k26 + H2I*HII*k16 + H2I*de*k17 + H2I*k31 - H2II*HI*k15 + H2II*Hm*k24 + H2II*de*k23 
    # + H2II*k28 + H2II*k30 - HI*HII*k14 - HI*HeI*k8 - HI*Hm*k13 - HI*de*k1 - HI*de*k12 - HI*k9 
    # + HII*Hm*k21 + HII*de*k2 + Hm*de*k19 + Hm*k27
    dnHIdt = -2*state[5]*state[0]*k26 + state[5]*state[1]*k16 + state[5]*state[8]*k17 
    + state[5]*k31 - state[6]*state[0]*k15 + state[6]*state[7]*k24 + state[6]*state[8]*k23
    + state[6]*k28 + state[6]*k30 - state[0]*state[1]*k14 - state[0]*state[2]*k14
    - state[0]*state[2]*k8 - state[0]*state[7]*k13 - state[0]*state[8]*k1 - state[0]*state[8]*k12
    - state[0]*k9 + state[1]*state[7]*k21 + state[1]*state[7]*k2 + state[7]*state[8]*k19
    + state[7]*k29
    
    # dnHIIdt = 
    # -H2I*HII*k16 + H2II*HI*k15 + H2II*k28 - HI*HII*k14 + HI*HeI*k8 + HI*de*k1 + 2*HI*k7 
    # + HI*k9 - HII*Hm*k21 - HII*Hm*k22 - HII*de*k2
    dnHIIdt = -state[5]*state[1]*k16 + state[6]*state[0]*k15 + state[6]*k28
    - state[0]*state[2]*k14 + state[0]*state[2]*k8 + state[0]*state[8]*k1
    + 2*state[0]*k7 + state[0]*k9 - state[1]*state[7]*k21 - state[1]*state[7]*k22
    - state[1]*state[8]*k2
    
    # dnHeIdt = -HeI*de*k3 - HeI*k10 + HeII*de*k4
    dnHeIdt = -state[2]*state[8]*k3 - state[2]*k10 + state[3]*state[8]*k4
    
    # dnHeIIdt = HeI*de*k3 + HeI*k10 - HeII*de*k4 - HeII*de*k5 - HeII*k11 + HeIII*de*k6
    dnHeIIdt = state[2]*state[8]*k3 + state[2]*k10 - state[3]*state[8]*k4
    - state[3]*state[8]*k5 - state[3]*k11 + state[4]*k6
    
    # dnHeIIIdt = HeII*de*k5 + HeII*k11 - HeIII*de*k6
    dnHeIIIdt = state[3]*state[8]*k3 + state[3]*k11 - state[4]*state[8]*k6
    
    # dnH2Idt = 
    # -H2I*HI*k18 - H2I*HII*k16 - H2I*de*k17 - H2I*k29 - H2I*k31 + H2II*HI*k15 + H2II*Hm*k24 + HI*Hm*k13 + 3*HI*k25
    dnH2Idt = -state[5]*state[0]*k18 - state[5]*state[1]*k16 - state[5]*state[8]*k17
    - state[5]*k29 - state[5]*k31 + state[6]*state[0]*k15 + state[6]*state[7]*k24
    + state[0]*state[7]*k13 + 3*state[0]*k25
    
    # dnH2IIdt = 
    # H2I*HII*k16 + H2I*k29 - H2II*HI*k15 - H2II*Hm*k24 - H2II*de*k23 - H2II*k28 - H2II*k30 + HI*HII*k14 + HII*Hm*k22
    dnH2IIdt = state[5]*state[1]*k16 + state[5]*k29 - state[6]*state[0]*k15 
    - state[6]*state[7]*k24 - state[6]*state[8]*k23 - state[6]*k28 - state[6]*k30
    + state[0]*state[1]*k14 + state[1]*state[7]*k22
    
    # dnHmdt = -H2II*Hm*k24 - HI*Hm*k13 - HI*Hm*k20 + HI*de*k12 - HII*Hm*k21 - HII*Hm*k22 - Hm*de*k19 - Hm*k27
    dnHmdt = -state[6]*state[7]*k24 - state[0]*state[7]*k13 - state[0]*state[7]*k20
    + state[0]*state[8]*k12 - state[1]*state[7]*k21 - state[1]*state[7]*k22
    - state[7]*state[8]*k19 - state[7]*k27
    
    # ddedt = 
    # H2I*k29 - H2II*de*k23 + H2II*k30 + HI*HeI*k8 + HI*Hm*k13 + HI*Hm*k20 - HI*de*k12 + 2*HI*k7 + HI*k9 
    # + HII*Hm*k22 - HII*de*k2 + HeI*k10 - HeII*de*k4 + HeII*k11 - HeIII*de*k6 + Hm*k27
    dndedt = state[5]*k29 - state[6]*state[8]*k23 + state[6]*k30 + state[0]*state[2]*k8
    + state[0]*state[7]*k13 + state[0]*state[7]*k20 - state[0]*state[8]*k12
    + 2*state[0]*k7 + state[0]*k9 + state[1]*state[7]*k22 - state[1]*state[8]*k2
    + state[2]*k10 - state[3]*state[8]*k4 + state[3]*k11 - state[4]*state[8]*k6
    + state[7]*k27
    
    return np.array([
        dnHIdt, dnHIIdt, dnHeIdt, dnHeIIdt, dnHeIIIdt, dnH2Idt, dnH2IIdt, dnHmdt, dndedt
    ])


# In[6]:


# Find reactions that form 'species'
def find_formation(species):
    rxns = []
    for r in all_reactions:
        if species in r[1].atoms():
            rxns.append(r)
    return rxns

# Find reactions that destroy 'species'
def find_destruction(species):
    rxns = []
    for r in all_reactions:
        if species in r[0].atoms():
            rxns.append(r)
    return rxns

# Get rhs of dS/dt equation to populate state vector
def get_rhs(species):
    dSdt = 0
    for lhs, rhs, coeff in find_formation(species):
        term = coeff
        for atom in list(lhs.atoms()):
            term *= atom
        dSdt += term
    for lhs, rhs, coeff in find_destruction(species):
        term = -coeff
        for atom in list(lhs.atoms()):
            term *= atom
        dSdt += term
    return dSdt


# In[7]:


# GIVEN:
# H, He, H2 ionization fractions (represented as 'f' in numDensity calculations)
# Gas density rho
# t_final
# integrator type ? 
# Mass fractions of H (0.74) and He (0.26)

# CALCULATIONS: 
# rho used to calculate number density for all of H where n(H_all) = rho * massFraction(H) / AtomicWeight(H)

# mu = totalMassDensity / totalNumberDensity
    # Note that totalMassDensity is in g/atom, need to change this to amu/atom 
    # totalNumberDensity = number densities of all ions 
    # massFraction(ion) = mass(ion)/mass(all)
    
# e = kT/(gamma-1)mu(atomicMass(H))
    # energy is constant
    # gamma = 5/3
    # solving for temperature, T 
    # initial T and initial mu give constant e value 
    
# --------------- STEPS ------------------------
    
# 1. Calculate total num density for H, He, H2 
#     n = rho * massFraction/AtomicWeight 
#     massFraction = massIon/massAllAtoms = numIon*massIon / sum(numElement*massElement) (don't have to calculate?)
    
# 2. Calculate individual number densities from ionization fractions and totalNumDensities from step 1

# 3. Sum all individual numDensities to get initial totalNumDensity 

# 4. Calculate initial mu (amu/atom) from rho and initial totalNumDensity 

# 5. Calculate initial e (constant) w/ initial T and mu

# 6. Initialize state vector and timestep vector 

# 5. for i = 0; i < t_final; i+=dt:
#     a. Do the integration 
#     b. Find individual number densities -> from state_vector_values (integrator.y)
#     c. Find totalNumDensity by summing values in integrator.y 
#     d. Calculate new mu 
#     e. Calculate new temperature, T, using e 
         


# In[8]:


# ----------- IMPLEMENTATION OF ABOVE PLAN (COMMENTED OUT BECAUSE IT DIDN'T WORK...) ---------
# # 1. Calculate total num density for H, He, H2 
#     # n = rho * massFraction/AtomicWeight 
#     # massFraction = massIon/massAllAtoms = numIon*massIon / sum(numElement*massElement) (don't have to calculate?)

# # Gas Density   
# rho = 1
# # Mass fractions
# f_H = 
# f_H2 = 
# f_He = 
# # Atomic Weights
# A_H = 
# A_He = 
# A_H2 = 2*A_H
# # Total Number Densities 
# n_H_all = rho * f_H/A_H
# n_He_all = rho * f_He/A_He
# n_H2_all = rho * f_H2/A_H2

# # 2. Calculate individual number densities from ionization fractions (f) and totalNumDensities from step 1
# n_H2 = (1 - f_H2) * n_H2_all
# n_H2I = f_H2 * n_H2_all
# n_H = (1 - f_H - 0.00001) * n_H_all 
# n_HI = f_H * n_H_all
# n_He = (1 - 0.00001 - f_He) * n_He_all
# n_HeI = f_He * n_He_all
# n_HeII = 0.00001 * n_He_all
# n_de = n_HI + n_He + 2*n_HeII - n_Hm + n_H2
# # n_Hm ?

# # 3. Sum all individual numDensities to get initial totalNumDensity 
# n_total = n_H2 + n_H2I + n_H + n_HI + n_He + n_HeI + n_HeII + n_de

# # 4. Calculate initial mu (amu/atom) from rho and initial totalNumDensity
# mu = rho / n_total

# # 5. Calculate e (constant) w/ initial T and mu
# boltzmann = 1.38064e-23 # m^2 kg s^-2 K^-1
# gamma = 5/3
# m_H = 1.00794 # amu 
# energy = (boltzmann * T) / ((gamma - 1)*mu*m_H)
    
    
    


# In[9]:


@ipywidgets.interact(n_total = (-3.0, 9.0), e_frac = (-8.0, 0.0),
                     T = (0., 6.), final_t = (0.0, 8.0),
                    safety_factor = (0, 10, 0.01))

def evolve(n_total = 2, e_frac = -4, T = np.log10(15000),
           final_t = 7,
           safety_factor = 10):
    
    final_t = 10**final_t  
    n_HI_initial = 10**n_total * (1.0 - 10**e_frac)
    n_HII_initial = 10**n_total * 10**e_frac
    n_HeI_initial = 10**n_total
    n_HeII_initial = 10**n_total * 10**e_frac
    n_HeIII_initial = 10**n_total
    n_H2I_initial = 10**n_total * 10**e_frac
    n_H2II_initial = 10**n_total
    n_Hm_initial = 10**n_total
    n_de_initial = 10**n_total * 10**e_frac
    
    # 6. Initialize state vector and timestep vector 
    state_vector = np.array([n_HI_initial
                             , n_HII_initial
                             , n_HeI_initial
                             , n_HeII_initial
                             , n_HeIII_initial
                             , n_H2I_initial
                             , n_H2II_initial
                             , n_Hm_initial
                             , n_de_initial])

    integrator = sint.ode(rhs)
    integrator.set_initial_value(state_vector, t=0)
    state_vector_values = []
    ts = [] #time step 
    
    # dt = safety_factor * np.min(state_vector/integrator.y)
    dt = final_t / safety_factor
        
    ts.append(integrator.t)
    state_vector_values.append(integrator.y)
    
    while integrator.t < final_t:
        time = integrator.t + dt
        # Integrate and update 
        integrator.integrate(time)
        ts.append(integrator.t)
        state_vector_values.append(integrator.y)
        # Update dt
        # dt = safety_factor * np.min(state_vector_values/integrator.y)
        # Prepare for next iteration - now things get weird 
#         n_total = sum(integrator.y) # New total number density from number densities that were just integrated
#         mu = rho / n_total
#         temperature = (energy*(gamma-1)*mu*m_H)/boltzmann

        
    state_vector_values = np.array(state_vector_values)
    ts = np.array(ts)
    
    plt.loglog(ts, state_vector_values[:,0], label='HI')
    plt.loglog(ts, state_vector_values[:,1], label='HII')
    plt.loglog(ts, state_vector_values[:,2], label='HeI')
    plt.loglog(ts, state_vector_values[:,3], label='HeII')
    plt.loglog(ts, state_vector_values[:,4], label='HeIII')
    plt.loglog(ts, state_vector_values[:,5], label='H2I')
    plt.loglog(ts, state_vector_values[:,6], label='H2II')
    plt.loglog(ts, state_vector_values[:,7], label='Hm')
    plt.loglog(ts, state_vector_values[:,8], label='de')
    
    plt.xlabel("Time [s]")
    plt.ylabel("n")
    plt.legend()

