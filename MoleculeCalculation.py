import numpy as np
from sympy.physics.quantum.cg import CG
import math
import scipy.linalg
import copy

import MarksConstants as mc
import Miscellaneous as misc

g = 0
u = 1

# ##################
# Creating Bases
# ##################
boStates = [
    {"L":1, "|Lambda|":0, "S":1, "I_BO":'g', "kappa_BO":1},
    {"L":1, "|Lambda|":0, "S":1, "I_BO":'u', "kappa_BO":1},
    {"L":1, "|Lambda|":0, "S":0, "I_BO":'g', "kappa_BO":1},
    {"L":1, "|Lambda|":0, "S":0, "I_BO":'u', "kappa_BO":1},
    {"L":1, "|Lambda|":1, "S":1, "I_BO":'g' },
    {"L":1, "|Lambda|":1, "S":1, "I_BO":'u' },
    {"L":1, "|Lambda|":1, "S":0, "I_BO":'g' },
    {"L":1, "|Lambda|":1, "S":0, "I_BO":'u' },
]

def addFsRelevantStates(boBasis):
    # Adds the states that are split by the Fs symmetry to the bare BO Basis.
    expandedBasis = []
    for state in boBasis:
        for Sigma in np.arange(-state['S'], state['S']+1,1):
            Omega = Sigma+state['|Lambda|']
            if state['|Lambda|'] != 0 and Omega==0: 
                # Pi state and Omega == 0 so Sigma = 1 and two ways of symmetrizing.
                for kappa_FS in [-1,1]:
                    newState = copy.copy(state)
                    newState.update({"|Sigma|":abs(Sigma), 'kappa_FS':kappa_FS, '|Omega|':abs(Omega)})
                    expandedBasis.append(newState)
            elif Sigma>=0:
                kappa_FS=(-1)**(1-state['|Lambda|']+state['S']-Sigma)
                newState = copy.copy(state)
                newState.update({"|Sigma|":abs(Sigma), 'kappa_FS':kappa_FS, '|Omega|':abs(Omega)})
                if newState not in expandedBasis:
                    expandedBasis.append(newState)
    return expandedBasis

def addHfsRelevantStates(boFsBasis, i_val=3/2):
    # Adds the states that are split by the Fs symmetry to the bare BO Basis.
    expandedBasis = []
    i_a = i_b = i_val
    IVals = np.arange(abs(i_a-i_b),i_a+i_b+1,1)
    for state in boFsBasis:
        for I_ in IVals:
            for Iota in np.arange(-I_, I_+1,1):
                #for Omega in [-state['|Omega|'], state['|Omega|']]:
                for Omega in [-state['|Omega|'], state['|Omega|']]:
                    Phi = Iota+Omega
                    mostUpdates = {"|Phi|":abs(Phi),'I':I_, '|Iota|':abs(Iota), 'i_a':i_a, 'i_b':i_b}
                    if Omega != 0 and Phi == 0:
                        # there are two ways of combining Omega and Iota to get Phi=0 then 
                        # and we will similarly get the symmetric and anti-symmetric superpositions.
                        for kappa_HFS in [-1,1]:
                            newState = copy.copy(state)
                            newState.update(mostUpdates)
                            newState.update({'kappa_HFS': kappa_HFS})
                            if newState not in expandedBasis:
                                expandedBasis.append(newState)
                    else:
                        newState = copy.copy(state)
                        newState.update(mostUpdates)
                        # probably want to think a moment on why using kappa_BO here
                        newState.update({ 'kappa_HFS':state['kappa_FS']*(-1)**(I_-Iota) })
                        if newState not in expandedBasis:
                            expandedBasis.append(newState)
    return expandedBasis

def createAtomicBases(lvals, svals, ivals):
    """
    Important notational note here. This function creates a *single atom* basis. 
    As such it is agnostic as to what nuclear center the basis is for, and so I 
    use an "x" subscript, which will be later changed to a/b when I construct the multi-particle basis. 
    Instead I continue to rely on that the notation of lower case letters is for a single particle quantum number
    and the upper case refer to joint quantum numbers (i.e. L = l_a + l_b). This _x business is just a mechanism 
    to construct the multiparticle basis.
    """
    lsiBasisRef, jiBasisRef, fBasisRef = [], [], []
    for l_ in lvals:
        for s_ in svals:
            jvals = set(np.arange(abs(s_ - l_), s_ + l_+1, 1))
            for j_ in jvals:
                for i_ in ivals:
                    fVals = set(np.arange(abs(j_ - i_), j_ + i_+1, 1))
                    for f_ in fVals:
                        for m_f in np.arange(-f_,f_+1,1):
                            fBasisRef.append(multiplyableDict({"f_x":f_, "m_f_x": m_f, "j_x":j_, "l_x":l_, "s_x":s_, "i_x":i_}))
                for m_j in np.arange(-j_,j_+1,1):
                    for i_ in ivals:
                        for m_i in np.arange(-i_,i_+1,1):
                            jiBasisRef.append(multiplyableDict({"j_x":j_, "m_j_x":m_j, "l_x":l_, "s_x":s_, "i_x":i_, "m_i_x":m_i}))    
        for m_l in np.arange(-l_,l_+1,1):
            for s_ in svals:
                for m_s in np.arange(-s_, s_+1,1):
                    for i_ in ivals:
                        for m_i in np.arange(-i_,i_+1,1):
                            lsiBasisRef.append(multiplyableDict({"l_x":l_, "m_l_x":m_l, "s_x":s_, "m_s_x":m_s, "i_x":i_, "m_i_x":m_i}))
    return (lsiBasisRef, jiBasisRef, fBasisRef, 
            np.kron(lsiBasisRef,lsiBasisRef), np.kron(jiBasisRef,jiBasisRef), np.kron(fBasisRef,fBasisRef))

def createCaseABasis_MostlySym(Lvals, Svals, ivals, I_BOvals=["g","u"], Jv=None, Fv=None):
    # Jv and Fv are only used for rotational calculations.
    # does not include the planar reflection symmetry kappa_HFS. 
    boBasisRef = []
    for I_BO in I_BOvals:
        for L_ in Lvals:
            for Lambda in np.arange(-L_,L_+1,1):
                for S_ in Svals:
                    for Sigma in np.arange(-S_, S_+1,1):
                        for i_a in ivals:
                            for i_b in ivals:
                                for I_ in np.arange(abs(i_b-i_a), i_b+i_a+1,1):
                                    for Iota in np.arange(-I_,I_+1,1):
                                        state = multiplyableDict({"L":L_, "Lambda": Lambda, 
                                                                  "I_BO": I_BO, "S":S_, "Sigma":Sigma,
                                                                  "I":I_, "Iota":Iota, "i_a":i_a, "i_b":i_b,
                                                                  "Omega":Sigma+Lambda, "Phi": Sigma+Lambda+Iota })
                                        if Jv is not None:
                                            state.update({'J':Jv})
                                        if Fv is not None:
                                            state.update({'F':Fv})
                                        if state not in boBasisRef:
                                            boBasisRef.append(state)
    boBasisRef = list(sorted(boBasisRef, key=lambda state: 1e5 * abs(state["Phi"]) 
                             + 1e2 * abs(state['Omega']) + 1 * abs(state['Sigma'])))
    return boBasisRef



def createCaseABasis_Sym(Lvals, Svals, ivals, I_BOvals=["g","u"], Jv=None, Fv=None):
    boBasisRef = []
    for I_BO in I_BOvals:
        for L_ in Lvals:
            for Lambda in np.arange(-L_,L_+1,1):
                for S_ in Svals:
                    for Sigma in np.arange(-S_, S_+1,1):
                        for i_a in ivals:
                            for i_b in ivals:
                                for I_ in np.arange(abs(i_b-i_a), i_b+i_a+1,1):
                                    for Iota in np.arange(-I_,I_+1,1):
                                        Omega = Sigma+Lambda
                                        Phi  = Sigma+Lambda+Iota
                                        if Lambda != 0 and Omega==0:
                                            for kappa_BO in [-1,1]:
                                                state = multiplyableDict({"L":L_, "|Lambda|": abs(Lambda), 
                                                                          "I_BO": I_BO, "S":S_, "|Sigma|":abs(Sigma),
                                                                          "I":I_, "|Iota|":abs(Iota), "i_a":i_a, "i_b":i_b,
                                                                          "|Omega|":abs(Omega), "Phi":Phi , 
                                                                          "kappa_BO": kappa_BO, 'kappa_HFS':kappa_BO*(-1)**(I_-Iota)})
                                                if Jv is not None:
                                                    state.update({'J':Jv})
                                                if Fv is not None:
                                                    state.update({'F':Fv})
                                                if state not in boBasisRef:
                                                    boBasisRef.append(state)
                                        elif Omega != 0 and Phi == 0:
                                            for kappa_HFS in [-1,1]:
                                                state = multiplyableDict({"L":L_, "|Lambda|": abs(Lambda), 
                                                                          "I_BO": I_BO, "S":S_, "|Sigma|":abs(Sigma), 
                                                                          "I":I_, "|Iota|":abs(Iota), "i_a":i_a, "i_b":i_b,
                                                                          "|Omega|":abs(Omega), "Phi":Phi , 
                                                                          "kappa_BO": (-1)**(L_-Lambda+S_-Sigma), 'kappa_HFS':kappa_HFS})
                                                if Jv is not None:
                                                    state.update({'J':Jv})
                                                if Fv is not None:
                                                    state.update({'F':Fv})
                                                if state not in boBasisRef:
                                                    boBasisRef.append(state)
                                        else:
                                            state = multiplyableDict({"L":L_, "|Lambda|": abs(Lambda), 
                                                                      "I_BO": I_BO, "S":S_, "|Sigma|":abs(Sigma), 
                                                                      "I":I_, "|Iota|":abs(Iota), "i_a":i_a, "i_b":i_b,
                                                                      "|Omega|":abs(Omega), "Phi": Phi, 
                                                                      "kappa_BO": (-1)**(L_-Lambda+S_-Sigma), 
                                                                      "kappa_HFS":(-1)**(L_-Lambda+S_-Sigma+I_-Iota) })
                                            if Jv is not None:
                                                state.update({'J':Jv})
                                            if Fv is not None:
                                                state.update({'F':Fv})
                                            if state not in boBasisRef:
                                                boBasisRef.append(state)
    boBasisRef = list(sorted(boBasisRef, key=lambda state: abs(state["|Omega|"])))
    return boBasisRef

# ######################################
# Basis Conversion Functions
# ######################################

def convertH_toCaseABasis(states, H_, offset=-1/2):
    # this seems misnamed...
    num = len(states)
    coupleM = np.array([[0.0 for _ in states] for _ in states])
    for num, _ in enumerate(states):
        coupleM[num,num] = offset
    # This seems like a very round-about way of doing this. 
    # states on the input is the conversino of a case-a state to the given H_'s base. So it's
    # |fs><a|
    for num1, state1 in enumerate(states):
        #misc.reportProgress(num1, len(states))
        for num2, state2 in enumerate(states):
            matElem = state2.T@H_@state1
            coupleM[num1,num2] += matElem
    return coupleM

def caseASymHfsToMostlySym(state, mostlySymBasis, indexes=False):
    # this is one of my weird transformations that I want to revise to be a normal matrix.         
    if state['|Omega|'] == 0 and state['|Iota|'] == 0:
        return caseASymFsToMostlySym(state, mostlySymBasis, indexes=indexes)
    else:
        stateMostlySym1, stateMostlySym2 = {},{}        
        for key in state.keys():
            if key in ["kappa_BO", "kappa_FS", "kappa_HFS", "|Lambda|", "|Sigma|", "|Omega|"]:
                pass
            elif key[0] != "|":
                stateMostlySym1[key] = state[key]
                stateMostlySym2[key] = state[key]
            elif key == "|Phi|":
                stateMostlySym1["Phi"] = state["|Phi|"]
            elif key == "|Iota|":
                if (state['|Iota|'] + state['|Omega|'] == state["|Phi|"]) or (state['|Iota|'] - state['|Omega|'] == state["|Phi|"]):
                    IotaSign = 1
                else:
                    IotaSign = -1
                stateMostlySym1["Iota"] = IotaSign*state["|Iota|"]
        
        stateMostlySym1['Omega'] = stateMostlySym1['Phi'] - stateMostlySym1['Iota']
        stateMostlySym1['Lambda'] = (1 if stateMostlySym1['Omega']>0 else -1)*state['|Lambda|']
        stateMostlySym1['Sigma'] = stateMostlySym1['Omega'] - stateMostlySym1['Lambda']
        for key in ['Phi','Iota','Omega','Lambda','Sigma']:
            stateMostlySym2[key] = -stateMostlySym1[key]        
        # Im confused about why this seems to need to involve kappa_FS to work.
        sign = '+' if state['kappa_HFS']*state['kappa_FS']*(-1)**(state['I']-state['|Iota|'])==1 else '-'
        #sign = '+' if state['kappa_HFS']*(-1)**(state['I']-state['|Iota|'])==1 else '-'
        #sign = '+' if state['kappa_HFS'] == 1 else '-'
        if indexes:
            return [mostlySymBasis.index(stateMostlySym1), mostlySymBasis.index(stateMostlySym2)], [1,1 if sign == "+" else -1]
        return '|'+''.join([str(val) for key, val in stateMostlySym1.items()])+'>'+sign+'|'+''.join([str(val) for key, val in stateMostlySym2.items()])+'>'
    
def caseASymFsToMostlySym(state, mostlySymBasis, indexes=False):
    # this is one of my weird transformations that I want to revise to be a normal matrix. 
    # I think this is the last one for the fine structure... gah of course the most tricky too.
    if state['|Lambda|'] == 0 and state['|Sigma|'] == 0:
        stateMostlySym = {}
        for key in state.keys():
            if key in ["kappa_BO", "kappa_FS", "kappa_HFS"]:
                pass
            elif key == '|Iota|':
                stateMostlySym['Iota'] = state['|Phi|']
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
            if key in ["kappa_BO", "kappa_FS", "kappa_HFS"]:
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
            elif key == "|Phi|":
                stateMostlySym1["Phi"] = state["|Phi|"]
                stateMostlySym2["Phi"] = -state["|Phi|"]
            elif key == "|Iota|":
                stateMostlySym1["Iota"] = state["|Phi|"]-state["|Omega|"]
                stateMostlySym2["Iota"] = state["|Phi|"]-(-state["|Omega|"])
        sign = '+' if state['kappa_FS'] == 1 else '-'
        #sign = '+' if state['kappa_BO'] == 1 else '-'
        if indexes:
            return [mostlySymBasis.index(stateMostlySym1), mostlySymBasis.index(stateMostlySym2)], [1,1 if sign == "+" else -1]
        return '|'+''.join([str(val) for key, val in stateMostlySym1.items()])+'>'+sign+'|'+''.join([str(val) for key, val in stateMostlySym2.items()])+'>'

def create_lsiToJi_Op(lsiBasis, jiBasis):
    """
    creates | j m_j i m_i > < l m_l s m_s i m_i | transformation matrix. 
    The matrix elements are just clebsch Gordon coefficeints, but you have
    to be careful to track quantum numbers carefully. 
    expects lsiBasis and jiBasis to be single atom bases.
    """
    assert(len(lsiBasis)==len(jiBasis))
    op = np.zeros((len(lsiBasis),len(jiBasis)))
    for lsnum, lsiState in enumerate(lsiBasis):
        for jnum, jiState in enumerate(jiBasis):
            # there should be some repeats because of the i values in each basis
            l, m_l, s, m_s, i_lsi, m_i_lsi = [lsiState[key] for key in ['l_x','m_l_x','s_x','m_s_x', 'i_x', 'm_i_x']]
            j, m_j, jl, js, i_ji, m_i_ji = [jiState[key] for key in ['j_x', 'm_j_x', 'l_x', 's_x', 'i_x', 'm_i_x']]
            # a good example of where you really need to keep track of all the quantum numbers. 
            # needing to handle this case makes me feel like the actual clebsh gordon coef should be written as
            # <L,mL,S,mS|J,mJ,l_b,s_b> or so instead of <L,mL,S,mS|J,mJ> as it usually is written. 
            if jl != l or js != s or i_lsi != i_ji or m_i_lsi != m_i_ji:
                op[jnum,lsnum] = 0
            else:
                op[jnum,lsnum] += float(CG(l, m_l, s, m_s, j, m_j).doit())
    return op

def create_lsi2ToJi2_Op(lsiBasis, jiBasis):
    """
    creates | j_a m_j_a i_a m_i_a >| j_b m_j_b i_b m_i_b > < l_a m_l_a s_a m_s_a i_a m_i_a |< l_b m_l_b s_b m_s_b i_b m_i_b | 
    transformation matrix. The matrix elements are just clebsch Gordon coefficeints, but you have
    to be careful to track quantum numbers carefully. 
    expects lsiBasis and jiBasis to be *two* atom bases.
    """
    assert(len(lsiBasis)==len(jiBasis))
    op = np.zeros((len(lsiBasis),len(jiBasis)))
    for lsnum, lsiState in enumerate(lsiBasis):
        for jnum, jiState in enumerate(jiBasis):
            op[jnum,lsnum] = 1
            for suffix in ['_a','_b']:
                # there should be some repeats because of the i values in each basis
                l, m_l, s, m_s, i_lsi, m_i_lsi = [lsiState[key+suffix] for key in ['l','m_l','s','m_s', 'i', 'm_i']]
                j, m_j, jl, js, i_ji, m_i_ji = [jiState[key+suffix] for key in ['j', 'm_j', 'l', 's', 'i', 'm_i']]
                # a good example of where you really need to keep track of all the quantum numbers. 
                # needing to handle this case makes me feel like the actual clebsh gordon coef should be written as
                # <L,mL,S,mS|J,mJ,l_b,s_b> or so instead of <L,mL,S,mS|J,mJ> as it usually is written. 
                if jl != l or js != s or i_lsi != i_ji or m_i_lsi != m_i_ji:
                    op[jnum,lsnum] *= 0
                else:
                    op[jnum,lsnum] *= float(CG(l, m_l, s, m_s, j, m_j).doit())
    return op

def create_jiToF_Op(jiBasis, fBasis):
    """
    creates the matrix |f m_f j i><j m_j i m_i| transformation matrix for the given bases.
    expects single atom bases
    """    
    assert(len(jiBasis)==len(fBasis))
    jiToF = np.zeros((len(jiBasis),len(fBasis)))
    for jnum, jiState in enumerate(jiBasis):
        for fnum, fState in enumerate(fBasis):
            j, m_j, l, s, i, m_i = [jiState[key] for key in ['j_x','m_j_x', 'l_x', 's_x', 'i_x', 'm_i_x']]
            f, m_f, fj, fi, fl, fs = [fState[key] for key in ['f_x','m_f_x','j_x','i_x', 'l_x', 's_x']]
            # needing to handle this case makes me feel like the actual clebsh gordon coef 
            # should technically be written as <L,mL,S,mS|J,mJ,l_b,s_b> or so instead 
            # of <L,mL,S,mS|J,mJ> as it usually is written. 
            if fj != j or fi != i or fl != l or fs != s:
                jiToF[fnum, jnum] = 0
            else:
                res = float(CG(j, m_j, i, m_i, f, m_f).doit())
                jiToF[fnum, jnum] += float(CG(j, m_j, i, m_i, f, m_f).doit())
    return jiToF

def genCaseAToLsiTransform(caseABasis, lsiBasis,caseAMostlySymHfs, basisChange=None):
    caseAToLsi = np.zeros((len(caseABasis),len(lsiBasis)**2))
    for staten, state in enumerate(caseABasis):        
        misc.reportProgress(staten, len(caseABasis))
        indexes, signs = caseASymHfsToMostlySym(state, caseAMostlySymHfs, indexes=True)
        symFsState = 0
        # construct the correct superposition which will preserve the symmetry of the given base. 
        # This should be only either 1 or 2 iterations in this loop.
        print(indexes, signs)
        if state['|Omega|'] != 0:
            indexes = [indexes[0]]
            signs = [signs[0]]
        for index, sign in zip(indexes,signs):
            caseAState = caseAMostlySymHfs[index] # alias
            I_BO = g if caseAState["I_BO"] == "g" else u
            #nuclearNums = [caseAState['I'], caseAState['Iota'],caseAState['i_a'],caseAState['i_b']]
            oalNums = (caseAState["L"], caseAState["Lambda"], 1, 0) # oal = "orbital angular momentum"
            spinNums = (caseAState["S"], caseAState["Sigma"], 1/2, 1/2)
            symFsState += sign * caseAToAtomic( oalNums, spinNums, (0,0,0,0), I_BO, lsiBasis, basisChange=basisChange )
        caseAToLsi[staten, :] = symFsState[:,0] / np.sqrt(len(indexes));
    return caseAToLsi

def caseAMostlySymToLsi_2Transf( caseABasis, lsiBasis2, verbose=False):
    """
    A hopefully more sensible way to calculate this. Should be able to make a matrix and then 
    """
    print('starting caseAMostlySymToLsi_2Transf')
    transformation = np.zeros((len(caseABasis), len(lsiBasis2)))
    for stateA_n, stateA in enumerate(caseABasis):
        L_, Lambda = [stateA[kv] for kv in ['L','Lambda']]
        l_a_sa, l_b_sa = (1,0)
        S_, Sigma = [stateA[kv] for kv in ['S','Sigma']]
        s_a_sa, s_b_sa = (1/2,1/2)
        I_, Iota, i_a_sa, i_b_sa = [stateA[kv] for kv in ['I','Iota','i_a','i_b']]
        I_BO = 1 if stateA['I_BO']=='g' else -1
        p_ = (-1)**(S_)*I_BO
        if verbose:
            print('pval',p_)
        for stateLsi_n, stateLsi in enumerate(lsiBasis2):
            (l_a, lambda_a, l_b, lambda_b) = [stateLsi[kv] for kv in ['l_a', 'm_l_a', 'l_b', 'm_l_b']]
            (s_a, sigma_a, s_b, sigma_b)   = [stateLsi[kv] for kv in ['s_a', 'm_s_a', 's_b', 'm_s_b']]
            (i_a, iota_a, i_b, iota_b)     = [stateLsi[kv] for kv in ['i_a', 'm_i_a', 'i_b', 'm_i_b']]
            if (S_ not in np.arange(abs(s_a-s_b), s_a+s_b+1,1) 
                or I_ not in np.arange(abs(i_a-i_b), i_a+i_b+1,1)
                or Lambda != lambda_a+lambda_b
                or Sigma != sigma_a + sigma_b
                or Iota != iota_a + iota_b
                or l_a == l_b):
                #print('continuing...')
                continue
            d1 = 1 if s_a_sa == s_a else 0
            d2 = 1 if s_b_sa == s_b else 0
            d3 = 1 if i_a_sa == i_a else 0
            d4 = 1 if i_b_sa == i_b else 0
            deltas = d1*d2*d3*d4
            oalCoef = float(CG(l_a,lambda_a,l_b,lambda_b,L_,Lambda).doit())
            spinCoef = float(CG(s_a,sigma_a,s_b,sigma_b,S_,Sigma).doit())
            nuclearCoef = float(CG(i_a,iota_a,i_b,iota_b,I_,Iota).doit())            
            transformation[stateA_n, stateLsi_n] = p_**(l_a)*deltas*oalCoef*spinCoef*nuclearCoef

        if np.linalg.norm(transformation[stateA_n,:]) == 0:
            raise ValueError("State has zero norm!",stateA,stateA_n)
        transformation[stateA_n,:] /= np.linalg.norm(transformation[stateA_n,:])
    return transformation

def genCaseAToLsiTransform2(caseABasis, lsiBasis, caseAMostlySymHfs, basisChange=None):
    print('starting genCaseAToLsiTransform2')
    caseAToLsi = np.zeros((len(caseABasis),len(lsiBasis)))
    transf = caseAMostlySymToLsi_2Transf( caseAMostlySymHfs, lsiBasis, verbose=False)
    
    for staten, state in enumerate(caseABasis):        
        misc.reportProgress(staten, len(caseABasis))
        indexes, signs = caseASymHfsToMostlySym(state, caseAMostlySymHfs, indexes=True)
        symFsState = 0
        # construct the correct superposition which will preserve the symmetry of the given base. 
        # This should be only either 1 or 2 iterations in this loop.
        print(indexes, signs)
        if state['|Omega|'] != 0:
            indexes = [indexes[0]]
            signs = [signs[0]]
        for index, sign in zip(indexes,signs):
            caseAState = caseAMostlySymHfs[index]
            caseAc = getColumnState(caseAMostlySymHfs, caseAState)
            print(transf.shape, np.array(caseAc).shape)
            symFsState += sign * transf.T @ caseAc
        caseAToLsi[staten, :] = symFsState[:,0] / np.sqrt(len(indexes));
    return caseAToLsi


def caseAToLsi_2Transf( caseABasis, lsiBasis2 ):
    """
    A hopefully more sensible way to calculate this. Should be able to make a matrix and then 
    """
    (L, Lambda, la, lb) = oalNums
    (S, Sigma, sa, sb) = spinNums
    (I, Iota, ia, ib) = nuclearNums
    p_ = (-1)**(S+I_BO)
    transformation = np.zeros((len(caseABasis, lsiBasis2)))
    for stateA_n, stateA in enumerate(caseABasis):
        for stateLsi_n, stateLsi in enumerate(lsiBasis2):
            (l_a, lambda_a, l_b, lambda_b) = [stateLsi[kv] for kv in ['l_a', 'm_l_a', 'l_b', 'm_l_b']]
            (s_a, sigma_a, s_b, sigma_b)   = [stateLsi[kv] for kv in ['s_a', 'm_s_a', 's_b', 'm_s_b']]
            (i_a, iota_a, i_b, iota_b)     = [stateLsi[kv] for kv in ['i_a', 'm_i_a', 'i_b', 'm_i_b']]
            
            oalCoef = float(CG(l_a,lambda_a,l_b,lambda_b,L,Lambda).doit())
            spinCoef = float(CG(s_a,sigma_a,s_b,sigma_b,S,Sigma).doit())
            nuclearCoef = float(CG(i_a,iota_a,i_b,iota_b,I,Iota).doit())
            
            transformation[stateA_n, stateLsi_n] = p_**(l_b)*oalCoef*spinCoef*nuclearCoef
            
    return transformation
            

def caseAToAtomic( oalNums, spinNums, nuclearNums, I_BO, lsiBasis, basisChange=None ):
    """
    This takes a single case A *state* and transforms it to the LSI basis an returns it. a rather round-about way to do this.
    
    (L, Lambda, la, lb) = oalNums
    (S, Sigma, sa, sb) = spinNums
    (I, Iota, ia, ib) = nuclearNums
    first converts to lsi basis then to whatever basis determined by basischange.
    
    this is the place to focus next.
    """
    
    state = 0
    (L, Lambda, la, lb) = oalNums
    (S, Sigma, sa, sb) = spinNums
    (I, Iota, ia, ib) = nuclearNums
    p_ = (-1)**(S+I_BO)
    for mla in np.arange(-la,la+1,1):
        mlb = Lambda-mla
        if abs(mlb) > lb:
            continue
        for msa in np.arange(-sa,sa+1,1):
            msb = Sigma-msa
            if abs(msb) > sb:
                continue
            for mia in np.arange(-ia, ia+1, 1):
                mib = Iota-mia
                if abs(mib) > ib:
                    continue
                # for mib in np.arange(-ib,ib+1,1):
                # CG notation is <j_a,mj_a,j_b,mj_b|j3,mj3>
                oalCoef = float(CG(la,mla,lb,mlb,L,Lambda).doit())
                spinCoef = float(CG(sa,msa,sb,msb,S,Sigma).doit())
                nuclearCoef = float(CG(ia,mia,ib,mib,I,Iota).doit())
                # I need to remember how this is organized...
                state1_a = getColumnState(lsiBasis, {'l_x':la,'m_l_x':mla,'s_x':sa,'m_s_x':msa, 'i_x':ia,'m_i_x':mia})
                state1_b = getColumnState(lsiBasis, {'l_x':lb,'m_l_x':mlb,'s_x':sb,'m_s_x':msb, 'i_x':ib,'m_i_x':mib})
                state2_a = getColumnState(lsiBasis, {'l_x':lb,'m_l_x':mlb,'s_x':sa,'m_s_x':msa, 'i_x':ia,'m_i_x':mia})
                state2_b = getColumnState(lsiBasis, {'l_x':la,'m_l_x':mla,'s_x':sb,'m_s_x':msb, 'i_x':ib,'m_i_x':mib})
                if basisChange is not None:
                    state1_a = basisChange @ state1_a
                    state1_b = basisChange @ state1_b
                    state2_a = basisChange @ state2_a
                    state2_b = basisChange @ state2_b
                newpart = nuclearCoef*oalCoef*spinCoef * (np.kron(state1_a,state1_b) + p_ * np.kron(state2_a,state2_b))
                state += newpart
    if np.linalg.norm(state) == 0:
        raise ValueError("State has zero norm!")
    state /= np.linalg.norm(state)
    return state

# #####################################
# Original Hamiltonian Creation
# #####################################

def create_H_HFS(hfs_basis, E_5P12_F1F2_splitting, E_HFS_5S12_F1F2_splitting, F3E, F2E, F1E, F0E):
    # hfs_basis: expects a two-particle basis, so each element of the basis should 
    # have Fz, mF_a, J_a, i_a, and F_b, mF_b, J_b, i_b values.
    # A_5P12_F1F2: The D1 line excited state hyperfine splitting (energy between 5P_{1/2},F=1 and F=2)
    # F3E, F2E, F1E, F0E: the energies of the 5P_{3/2} F=3,2,1,0 manifolds. In this case, unlike in 
    # the fine-structure case, because there are four energy levels perturbations to the energy levels 
    # can't be captured in a single "A" constant, so instead I use the actual energy values 
    # (F3E, F2E, F1E, F0E)
    H_HFS = np.zeros((len(hfs_basis),len(hfs_basis)))
    #f_a,mf_a,j_a,i_a,f_b,mf_b,j_b,i_b = [0 for _ in range(8)]
    name_pref = ['f','m_f','j','i']
    for s1num, state1 in enumerate(hfs_basis):
        # unpack the values of the quantum numbers from the state:
        qNums = [0 for _ in range(8)]
        for num, name_p in enumerate(name_pref):
            qNums[num] = state1[name_p+"_a"]
            qNums[num+4] = state1[name_p+"_b"]
        f_a,mf_a,j_a,i_a,f_b,mf_b,j_b,i_b = qNums
        # calculate the energies of the individual atoms and add. A_a_HFS would be the hyperfine splitting constant (oftentimes denoted A_HFS) 
        # for particles around nucleus A
        A_a_HFS = E_5P12_F1F2_splitting if state1["l_a"] == 1 else E_HFS_5S12_F1F2_splitting
        E1 = A_a_HFS/2 * (f_a*(f_a+1)-j_a*(j_a+1)-i_a*(i_a+1)) if state1["j_a"] != 3/2 else (F3E if f_a==3 else (F2E if f_a==2 else (F1E if f_a==1 else F0E)))
        A_b_HFS = E_5P12_F1F2_splitting if state1["l_b"] == 1 else E_HFS_5S12_F1F2_splitting
        E2 = A_b_HFS/2 * (f_b*(f_b+1)-j_b*(j_b+1)-i_b*(i_b+1)) if state1["j_b"] != 3/2 else (F3E if f_b==3 else (F2E if f_b==2 else (F1E if f_b==1 else F0E)))
        # the matrix is diagonal in the given basis.
        H_HFS[s1num,s1num] = E1 + E2
    return H_HFS

def create_H_FS(fs_basis):
    # expects a two-particle basis, so each element of the basis should 
    # have j_a, mj_a, l_a, s_a, and J_b, mJ_b, l_b, s_b values.
    H_FS = np.zeros((len(fs_basis), len(fs_basis)))
    #j_a,mj_a,l_a,s_a,j_b,mj_b,l_b,s_b = [0 for _ in range(8)]
    names = ['j','m_j','l','s']
    # it's diagonal in this fs_basis, so only one loop.
    for s1num, state1 in enumerate(fs_basis):
        qNums = [0 for _ in range(8)]
        for num, name in enumerate(names):
            qNums[num] = state1[name+"_a"]
        for num, name in enumerate(names):
            qNums[num+4] = state1[name+"_b"]
        j_a,m_j_a,l_a,s_a,j_b,m_j_b,l_b,s_b = qNums
        # the matrix element is A/2 (l_a . s_a + l_b . s_b), A=1
        val = 0.5 * (  j_a*(j_a+1)-l_a*(l_a+1)-s_a*(s_a+1) 
                     + j_b*(j_b+1)-l_b*(l_b+1)-s_b*(s_b+1))
        H_FS[s1num,s1num] = val
    return H_FS

def get_H_BO(C3, Rv, bo_basis):
    # returns the Born Oppenheimer Hamiltonian
    # expects states to be a list of lists where the low level list has the values of L, Lambda, S,Sigma, and I_BO.
    #H_BO = np.array([[0.0 for _ in bo_basis] for _ in bo_basis])
    H_BO = np.zeros((len(bo_basis),len(bo_basis)))
    lambdaKey = "|Lambda|" if "|Lambda|" in bo_basis[0] else "Lambda"
    # the matrix is diagonal.
    for num, state in enumerate(bo_basis):
        I_BO = g if state["I_BO"]=="g" else u
        pv = (-1)**(state["S"]+I_BO)
        L_ = state["L"]
        H_BO[num,num] = -pv*(3*state[lambdaKey]**2-L_*(L_+1))/Rv**3 * C3
    return H_BO

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
            key_ag = key[:-2] # remove the _x from the single particle agnostic labels
            newDict.update({key_ag+"_a": value})
        for key, value in other.items():
            key_ag = key[:-2]
            newDict.update({key_ag+"_b": value})
        return newDict

def getColumnState(basis, quantumNums):
    # Mostly I use bases in their dictionary form where each element has all the 
    # relevant quantum numbers readily accessibly to identify the state. But eventually
    # I need to calculate matrices which are indexed by numbers, not a list of quantum
    # numbers. This function helps facilitate the relationship between these.
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
        if key != 'L' and key != 'Phi' and key != 'i_a' and key != 'i_b':
            label += key + ":" + str(val) + ", "
    return label
