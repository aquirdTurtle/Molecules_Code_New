import numpy as np
from sympy.physics.quantum.cg import CG
import math
import scipy.linalg
import MarksConstants as mc
import Miscellaneous as misc

g = 0
u = 1

# ##################
# Creating Bases
# ##################
    
def createAtomicBases(Lvals, Svals, Ivals):
    lsiBasisRef, jiBasisRef, fBasisRef = [], [], []
    for Lv in Lvals:
        for Sv in Svals:
            Jvals = set(np.arange(abs(Sv - Lv), Sv + Lv+1, 1))
            for Jv in Jvals:
                for Iv in Ivals:
                    FVals = set(np.arange(abs(Jv - Iv), Jv + Iv+1, 1))
                    for Fv in FVals:
                        for mf in np.arange(-Fv,Fv+1,1):
                            fBasisRef.append(multiplyableDict({"F":Fv, "mF": mf, "J":Jv, "L":Lv, "S":Sv, "I":Iv}))
                for mJ in np.arange(-Jv,Jv+1,1):
                    for Iv in Ivals:
                        for mi in np.arange(-Iv,Iv+1,1):
                            jiBasisRef.append(multiplyableDict({"J":Jv, "mJ":mJ, "L":Lv, "S":Sv, "I":Iv, "mI":mi}))    
        for mL in np.arange(-Lv,Lv+1,1):
            for Sv in Svals:
                for mS in np.arange(-Sv, Sv+1,1):
                    for Iv in Ivals:
                        for mi in np.arange(-Iv,Iv+1,1):
                            lsiBasisRef.append(multiplyableDict({"L":Lv, "mL":mL, "S":Sv, "mS":mS, "I":Iv, "mI":mi}))
    return (lsiBasisRef, jiBasisRef, fBasisRef, 
            np.kron(lsiBasisRef,lsiBasisRef), np.kron(jiBasisRef,jiBasisRef), np.kron(fBasisRef,fBasisRef))

def createCaseABasis_MostlySym(Lvals, Svals, Ivals, sigmavals=["g","u"], Jv=None):
    # Jv is only used for rotational calculations
    # does not include the planar reflection symmetry sigma_v2. 
    boBasisRef = []
    for sigma in sigmavals:
        for Lv in Lvals:
            for Lambda in np.arange(-Lv,Lv+1,1):
                for Sv in Svals:
                    for Sigma in np.arange(-Sv, Sv+1,1):
                        for i1 in Ivals:
                            for i2 in Ivals:
                                for I in np.arange(abs(i2-i1), i2+i1+1,1):
                                    for iota in np.arange(-I,I+1,1):
                                        state = multiplyableDict({"L":Lv, "Lambda": Lambda, 
                                                                  "sigma": sigma, "S":Sv, "Sigma":Sigma,
                                                                  "i":I, "iota":iota, "i1":i1, "i2":i2,
                                                                  "Omega":Sigma+Lambda, "phi": Sigma+Lambda+iota })
                                        if Jv is not None:
                                            state.update({'J':Jv})
                                        if state not in boBasisRef:
                                            boBasisRef.append(state)
    boBasisRef = list(sorted(boBasisRef, key=lambda state: 1e5 * abs(state["phi"]) 
                             + 1e2 * abs(state['Omega']) + 1 * abs(state['Sigma'])))
    return boBasisRef



def createCaseABasis_Sym(Lvals, Svals, Ivals, sigmavals=["g","u"], Jv=None):
    boBasisRef = []
    for sigma in sigmavals:
        for Lv in Lvals:
            for Lambda in np.arange(-Lv,Lv+1,1):
                for Sv in Svals:
                    for Sigma in np.arange(-Sv, Sv+1,1):
                        for i1 in Ivals:
                            for i2 in Ivals:
                                for Iv in np.arange(abs(i2-i1), i2+i1+1,1):
                                    for iota in np.arange(-Iv,Iv+1,1):
                                        Omega = Sigma+Lambda
                                        phi  = Sigma+Lambda+iota
                                        if Lambda != 0 and Omega==0:
                                            for sigmav in [-1,1]:
                                                state = multiplyableDict({"L":Lv, "|Lambda|": abs(Lambda), 
                                                                          "sigma": sigma, "S":Sv, "|Sigma|":abs(Sigma),
                                                                          "i":Iv, "|iota|":abs(iota), "i1":i1, "i2":i2,
                                                                          "|Omega|":abs(Omega), "phi":phi , 
                                                                          "sigma_vxz": sigmav, 'sigma_v2xz':sigmav*(-1)**(Iv-iota)})
                                                if Jv is not None:
                                                    state.update({'J':Jv})
                                                if state not in boBasisRef:
                                                    boBasisRef.append(state)
                                        elif Omega != 0 and phi == 0:
                                                for sigmav2 in [-1,1]:
                                                    state = multiplyableDict({"L":Lv, "|Lambda|": abs(Lambda), 
                                                                              "sigma": sigma, "S":Sv, "|Sigma|":abs(Sigma), 
                                                                              "i":Iv, "|iota|":abs(iota), "i1":i1, "i2":i2,
                                                                              "|Omega|":abs(Omega), "phi":phi , 
                                                                              "sigma_vxz": (-1)**(Lv-Lambda+Sv-Sigma), 'sigma_v2xz':sigmav2})
                                                    if Jv is not None:
                                                        state.update({'J':Jv})
                                                    if state not in boBasisRef:
                                                        boBasisRef.append(state)
                                        else:
                                            state = multiplyableDict({"L":Lv, "|Lambda|": abs(Lambda), 
                                                                      "sigma": sigma, "S":Sv, "|Sigma|":abs(Sigma), 
                                                                      "i":Iv, "|iota|":abs(iota), "i1":i1, "i2":i2,
                                                                      "|Omega|":abs(Omega), "phi": phi, 
                                                                      "sigma_vxz": (-1)**(Lv-Lambda+Sv-Sigma), 
                                                                      "sigma_v2xz":(-1)**(Lv-Lambda+Sv-Sigma+Iv-iota) })
                                            if Jv is not None:
                                                state.update({'J':Jv})
                                            if state not in boBasisRef:
                                                boBasisRef.append(state)
    boBasisRef = list(sorted(boBasisRef, key=lambda state: abs(state["|Omega|"])))
    return boBasisRef

# ######################################
# Basis Conversion Functions
# ######################################

def convertH_toCaseABasis(states, H_, offset=-1/2):
    num = len(states)
    coupleM = np.array([[0.0 for _ in states] for _ in states])
    for num, _ in enumerate(states):
        coupleM[num,num] = offset
    for num1, state1 in enumerate(states):
        #misc.reportProgress(num1, len(states))
        for num2, state2 in enumerate(states):
            matElem = state2.T@H_@state1
            coupleM[num1,num2] += matElem
    return coupleM    

def caseASymHfsToMostlySym(state, mostlySymBasis, indexes=False):
    if state['|Omega|'] == 0 and state['|iota|'] == 0:
        return caseASymFsToMostlySym(state, mostlySymBasis, indexes=indexes)
    else:
        stateMostlySym1, stateMostlySym2 = {},{}        
        for key in state.keys():
            if key == "sigma_vxz" or key == "sigma_v2xz" or key == "|Lambda|" or key == "|Sigma|" or key == "|Omega|":
                pass
            elif key[0] != "|":
                stateMostlySym1[key] = state[key]
                stateMostlySym2[key] = state[key]
            elif key == "phi":
                stateMostlySym1["phi"] = state["phi"]
            elif key == "|iota|":
                if (state['|iota|'] + state['|Omega|'] == state["phi"]) or (state['|iota|'] - state['|Omega|'] == state["phi"]):
                    iotaSign = 1
                else:
                    iotaSign = -1
                stateMostlySym1["iota"] = iotaSign*state["|iota|"]
        
        stateMostlySym1['Omega'] = stateMostlySym1['phi'] - stateMostlySym1['iota']
        stateMostlySym1['Lambda'] = (1 if stateMostlySym1['Omega']>0 else -1)*state['|Lambda|']
        stateMostlySym1['Sigma'] = stateMostlySym1['Omega'] - stateMostlySym1['Lambda']
        for key in ['phi','iota','Omega','Lambda','Sigma']:
            stateMostlySym2[key] = -stateMostlySym1[key]        
                
        # Im confused about why this seems to need to involve sigma_vxz to work.
        sign = '+' if state['sigma_v2xz']*state['sigma_vxz']*(-1)**(state['i']-state['|iota|'])==1 else '-'
        #sign = '+' if state['sigma_v2xz']*(-1)**(state['i']-state['|iota|'])==1 else '-'
        #sign = '+' if state['sigma_v2xz'] == 1 else '-'
        if indexes:
            return [mostlySymBasis.index(stateMostlySym1), mostlySymBasis.index(stateMostlySym2)], [1,1 if sign == "+" else -1]
        return '|'+''.join([str(val) for key, val in stateMostlySym1.items()])+'>'+sign+'|'+''.join([str(val) for key, val in stateMostlySym2.items()])+'>'
    
def caseASymFsToMostlySym(state, mostlySymBasis, indexes=False):
    if state['|Lambda|'] == 0 and state['|Sigma|'] == 0:
        stateMostlySym = {}
        for key in state.keys():
            if key == "sigma_vxz" or key == "sigma_v2xz":
                pass
            elif key == '|iota|':
                stateMostlySym['iota'] = state['phi']
            elif key[0] != "|":
                stateMostlySym[key] = state[key]
            else:
                stateMostlySym[key[1:-1]] = 0
        if indexes:
            return [mostlySymBasis.index(stateMostlySym)], [1]
        return '|'+''.join([str(val) for key, val in stateMostlySym.items()])+'>'
    else:
        # else two state contribute
        stateMostlySym2, stateMostlySym1 = {}, {}
        for key in state.keys():
            if key == "sigma_vxz" or key == "sigma_v2xz":
                pass
            elif key[0] != "|":
                stateMostlySym1[key] = state[key]
                stateMostlySym2[key] = state[key]
            elif key == "|Lambda|":
                stateMostlySym1[key[1:-1]] = state[key]
                stateMostlySym2[key[1:-1]] = -state[key]
            elif key == "|Sigma|":
                stateMostlySym1[key[1:-1]] = state["|Omega|"] - stateMostlySym1['Lambda']
                stateMostlySym2[key[1:-1]] = -state["|Omega|"] - stateMostlySym2['Lambda']
            elif key == "|Omega|":
                stateMostlySym1["Omega"] = state["|Omega|"]
                stateMostlySym2["Omega"] = -state["|Omega|"]
            elif key == "phi":
                stateMostlySym1["phi"] = state["phi"]
                stateMostlySym2["phi"] = -state["phi"]
            elif key == "|iota|":
                stateMostlySym1["iota"] = state["phi"]-state["|Omega|"]
                stateMostlySym2["iota"] = state["phi"]-(-state["|Omega|"])
        sign = '+' if state['sigma_vxz'] == 1 else '-'
        if indexes:
            return [mostlySymBasis.index(stateMostlySym1), mostlySymBasis.index(stateMostlySym2)], [1,1 if sign == "+" else -1]
        return '|'+''.join([str(val) for key, val in stateMostlySym1.items()])+'>'+sign+'|'+''.join([str(val) for key, val in stateMostlySym2.items()])+'>'

def create_lsiToJi_Op(lsiBasis, jiBasis):
    # expects single atom bases
    assert(len(lsiBasis)==len(jiBasis))
    op = np.zeros((len(lsiBasis),len(jiBasis)))
    for lsnum, lsiState in enumerate(lsiBasis):
        for jnum, jiState in enumerate(jiBasis):
            # there should be some repeats because of the i values in each basis
            L, mL, S, mS, I_lsi, mI_lsi = [lsiState[key] for key in ['L','mL','S','mS', 'I', 'mI']]
            J, mJ, JL, JS, I_ji, mI_ji = [jiState[key] for key in ['J','mJ', 'L', 'S', 'I', 'mI']]
            # needing to handle this case makes me feel like the actual clebsh gordon coef should be written as
            # <L,mL,S,mS|J,mJ,L2,S2> or so instead of <L,mL,S,mS|J,mJ> as it usually is written. 
            if JL != L or JS != S or I_lsi != I_ji or mI_lsi != mI_ji:
                op[jnum,lsnum] = 0
            else:
                op[jnum,lsnum] += float(CG(L, mL, S, mS, J, mJ).doit())
    return op

def create_jiToF_Op(jiBasis, fBasis):
    # expects single atom bases
    assert(len(jiBasis)==len(fBasis))
    op = np.zeros((len(jiBasis),len(fBasis)))
    for jnum, jiState in enumerate(jiBasis):
        for fnum, fState in enumerate(fBasis):
            J, mJ, L, S, I, mI = [jiState[key] for key in ['J','mJ', 'L', 'S', 'I', 'mI']]
            F, mF, Jf, If, Lf, Sf = [fState[key] for key in ['F','mF','J','I', 'L', 'S']]
            # needing to handle this case makes me feel like the actual clebsh gordon coef 
            # should technically be written as <L,mL,S,mS|J,mJ,L2,S2> or so instead 
            # of <L,mL,S,mS|J,mJ> as it usually is written. 
            if Jf != J or If != I or Lf != L or Sf != S:
                op[fnum, jnum] = 0
            else:
                res = float(CG(J, mJ, I, mI, F, mF).doit())
                op[fnum, jnum] += float(CG(J, mJ, I, mI, F, mF).doit())
    return op

def caseAToAtomic( oalNums, spinNums, nuclearNums, sigma, lsiBasis, basisChange=None ):
    state = 0
    otherBasisStates, lsiBasisStates, indvCont = [], [], []
    (L, Lambda, la, lb) = oalNums
    (S, Sigma, sa, sb) = spinNums
    (I, iota, ia, ib) = nuclearNums
    p_ = (-1)**(S+sigma)
    for mla in np.arange(-la,la+1,1):
        mlb = Lambda-mla
        if abs(mlb) > lb:
            continue
        for msa in np.arange(-sa,sa+1,1):
            msb = Sigma-msa
            if abs(msb) > sb:
                continue
            for mia in np.arange(-ia, ia+1, 1):
                mib = iota-mia
                if abs(mib) > ib:
                    continue
                # for mib in np.arange(-ib,ib+1,1):
                # CG notation is <j1,mj1,j2,mj2|j3,mj3>
                oalCoef = float(CG(la,mla,lb,mlb,L,Lambda).doit())
                spinCoef = float(CG(sa,msa,sb,msb,S,Sigma).doit())
                nuclearCoef = float(CG(ia,mia,ib,mib,I,iota).doit())
                aState1 = getColumnState(lsiBasis, {'L':la,'mL':mla,'S':sa,'mS':msa, 'I':ia,'mI':mia})
                bState1 = getColumnState(lsiBasis, {'L':lb,'mL':mlb,'S':sb,'mS':msb, 'I':ib,'mI':mib})
                aState2 = getColumnState(lsiBasis, {'L':lb,'mL':mlb,'S':sa,'mS':msa, 'I':ia,'mI':mia})
                bState2 = getColumnState(lsiBasis, {'L':la,'mL':mla,'S':sb,'mS':msb, 'I':ib,'mI':mib})
                if oalCoef != 0 and nuclearCoef != 0 and spinCoef != 0:
                    lsiBasisStates.append([aState1, bState1, aState2, bState2])
                if basisChange is not None:
                    aState1 = basisChange@aState1
                    bState1 = basisChange@bState1
                    aState2 = basisChange@aState2
                    bState2 = basisChange@bState2
                newpart = nuclearCoef*oalCoef*spinCoef * (np.kron(aState1,bState1) + p_ * np.kron(aState2,bState2))
                state += newpart
                if oalCoef != 0 and nuclearCoef != 0 and spinCoef != 0:
                    indvCont.append(newpart)
                    otherBasisStates.append([aState1, bState1, aState2, bState2])
    if np.linalg.norm(state) == 0:
        raise ValueError("State has zero norm!")
    state /= np.linalg.norm(state)
    return state #, np.array(indvCont), np.array(lsiBasisStates), np.array(otherBasisStates)

# #####################################
# Original Hamiltonian Creation
# #####################################

def create_HfsH(basis, E_5P12_F1F2_splitting, E_HFS_5S12_F1F2_splitting, F3E, F2E, F1E, F0E):
    # basis: expects a two-particle basis, so each element of the basis should 
    # have F_1, mF_1, J_1, I_1, and F_2, mF_2, J_2, I_2 values.
    # A_5P12_F1F2: The D1 line excited state hyperfine splitting (energy between 5P_{1/2},F=1 and F=2)
    # F3E, F2E, F1E, F0E: the energies of the 5P_{3/2} F=3,2,1,0 manifolds. In this case, unlike in 
    # the fine-structure case, because there are four energy levels perturbations to the energy levels 
    # can't be captured in a single "A" constant, so instead I use the actual energy values 
    # (F3E, F2E, F1E, F0E)
    op = np.zeros((len(basis),len(basis)))
    #f1,mf1,j1,i1,f2,mf2,j2,i2 = [0 for _ in range(8)]
    names = ['F','mF','J','I']
    for s1num, state1 in enumerate(basis):
        # unpack the actual values of the quantum numbers:
        qNums = [0 for _ in range(8)]
        for num, name in enumerate(names):
            qNums[num] = state1[name+"_1"]
            qNums[num+4] = state1[name+"_2"]
        #for num, name in enumerate(names):
        #    qNums[num+4] = state1[name+"_2"]
        f1,mf1,j1,i1,f2,mf2,j2,i2 = qNums
        # calculate the energies of the individual atoms and add.
        A1 = E_5P12_F1F2_splitting if state1["L_1"] == 1 else E_HFS_5S12_F1F2_splitting
        E1 = A1/2 * (f1*(f1+1)-j1*(j1+1)-i1*(i1+1)) if state1["J_1"] != 3/2 else (F3E if f1==3 else (F2E if f1==2 else (F1E if f1==1 else F0E)))
        A2 = E_5P12_F1F2_splitting if state1["L_2"] == 1 else E_HFS_5S12_F1F2_splitting
        E2 = A2/2 * (f2*(f2+1)-j2*(j2+1)-i2*(i2+1)) if state1["J_2"] != 3/2 else (F3E if f2==3 else (F2E if f2==2 else (F1E if f2==1 else F0E)))
        # the matrix is diagonal in the given basis.
        op[s1num,s1num] = E1 + E2
    return op

def create_fsH(basis):
    # expects a two-particle basis, so each element of the basis should 
    # have J_1, mJ_1, L_1, S_1, and J_2, mJ_2, L_2, S_2 values.
    op = np.zeros((len(basis),len(basis)))
    J1,mJ1,L1,S1,J2,mJ2,L2,S2 = [0 for _ in range(8)]
    names = ['J','mJ','L','S']
    for s1num, state1 in enumerate(basis):
        qNums = [0 for _ in range(8)]
        for num, name in enumerate(names):
            qNums[num] = state1[name+"_1"]
        for num, name in enumerate(names):
            qNums[num+4] = state1[name+"_2"]
        J1,mJ1,L1,S1,J2,mJ2,L2,S2 = qNums
        # the matrix element is A/2 (L1 . S1 + L2 . S2), A=1
        val = 0.5 * (  J1*(J1+1)-L1*(L1+1)-S1*(S1+1) 
                     + J2*(J2+1)-L2*(L2+1)-S2*(S2+1))
        op[s1num,s1num] = val
    return op

def getBoH(C3, Rv, *states):
    # returns the Born Oppenheimer Hamiltonian    
    # expects states to be a list of lists where the low level list has the values of L, Lambda, S,Sigma, and sigma.
    matrix = np.array([[0.0 for _ in states] for _ in states])
    # the matrix is diagonal.
    lambdaKey = "|Lambda|" if "|Lambda|" in states[0] else "Lambda"
    for num, state in enumerate(states):
        sigma = g if state["sigma"]=="g" else u
        pv = (-1)**(state["S"]+sigma)
        Lv = state["L"]
        matrix[num,num] = -pv*(3*state[lambdaKey]**2-Lv*(Lv+1))/Rv**3 * C3
    return matrix

# ##############
# Miscellaneous
# ##############

class multiplyableDict(dict):
    # This class exists so that I can take a basis ref and use it in np.kron()
    # to programatically get the basis ref for multi-particle systems.
    def __mul__(self, other):
        assert(type(other) == type(self))
        newDict = multiplyableDict()
        for key, value in self.items():
            newDict.update({key+"_1": value})
        for key, value in other.items():
            newDict.update({key+"_2": value})
        return newDict
    
def getColumnState(basis, quantumNums):
    assert(len(basis[0])==len(quantumNums))
    colState = [[0] for _ in range(len(basis))]
    for num, state in enumerate(basis):
        match = True
        for qnum, val in quantumNums.items():
            if val != state[qnum]:
                match = False
        if match:
            colState[num][0] = 1
            return colState
    raise ValueError("No Match! nums were" + str(quantumNums))
    
def stateLabel(state):
    label = ""
    for key, val in state.items():
        if key != 'L' and key != 'phi' and key != 'i1' and key != 'i2':
            label += key + ":" + str(val) + ", "
    return label
