import os


# compute cost for each stage of framework (MVCNN)

A = 180*3783/3600
print(A)
components = ['Flow', 'Decode', 'tstream', 'sstream']

# p2 instance price
F_list = []
for comp in components:
    F_list.append(float(input('Input {} FPS:'.format(comp))))

flow_check = input('Is flow computed on a GPU?')
print(F_list)

C_list = []
for i,comp in enumerate(components):
    if (flow_check.lower() == 'n' and comp == 'Flow') or comp == 'Decode':
        P = 0.333
    else:
        P = 0.9
    print(P)
    C_list.append(A*P/F_list[i])

print(C_list)
C_tot = (sum(C_list))
print(C_tot)

#print('Total cost: {}, Flow cost: {}, Decode cost: {}, Tstream cost: {}, Sstream cost {}'.format(C_tot, *C_list))