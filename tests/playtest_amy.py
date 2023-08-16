import numpy as np
# import sympy as sp

# from hierarqcal import (canonical_reshape, contract_tensors)

# e_2=(np.array([1, 0]), np.array([0, 1]))
# e_3=(np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))

# CN_m = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
# CN_m = canonical_reshape(CN_m)
# eye = canonical_reshape(np.eye(2,2))
# eye_2 = canonical_reshape(np.kron(np.eye(2,2),np.eye(2,2)))

# x_0 = contract_tensors(CN_m, eye, [1,3],[0, 1])
# x_1 = contract_tensors(CN_m, eye, [1],[0])
# x_2 = contract_tensors(CN_m, eye, [0],[1])
# print((canonical_reshape(np.array([[2,0,0,0]])) == x_0).all())
# print((x_1 == CN_m).all()) 
# print((x_2 == CN_m).all()) 

# x_1 = contract_tensors(CN_m, eye, [1],[0], circuit = False)
# x_2 = contract_tensors(CN_m, eye, [0],[1], circuit = False)
# print((np.moveaxis(x_1,[3],[1]) == CN_m).all()) 
# print((np.moveaxis(x_2,[3],[0]) == CN_m).all()) 

# x_3 = contract_tensors(CN_m, eye_2, [1,2],[0,1])
# print((x_3 == CN_m).all()) 

# N = 5
# random = np.random.rand(*[2 for _ in range(N)])
# eye = np.eye(2)
# for i in range(N-1):
#     eye = np.kron(eye,np.eye(2))
# eye = canonical_reshape(eye)
# x = contract_tensors(random, eye)
# print(( x == random ).all())

# print()


###############################################

##############################################
# import timeit
  
# number_of_times = 10*5
# number_of_repeats = 10
  
# timeit_dictionary = {
#   0: {
#      "name": "move axis",
#      "setup": "from hierarqcal import (canonical_reshape, contract_tensors, test_func)",
#      "statement": "contract_tensors_1(*test_func())"
#   },
#   1: {
#      "name": "transpose",
#      "setup": "from hierarqcal import (canonical_reshape, contract_tensors, test_func)",
#      "statement": "contract_tensors_2(*test_func())"
#     },
# }

# for id, library in timeit_dictionary.items():
#   print(f"Time it takes to run different version of contract")
#   result = timeit.repeat(stmt=library["statement"],
#      setup=library["setup"],
#      number=number_of_times,
#      repeat=number_of_repeats)
#   print(np.mean(result))


###############################################

##############################################
import timeit
  
number_of_times = 3
number_of_repeats = 3
  
timeit_dictionary = {
  0: {
     "name": "",
     "setup": "from hierarqcal import (test_func)",
     "statement": "test_func(1)"
  },
  1: {
     "name": "",
     "setup": "from hierarqcal import (test_func)",#+"\n"+"global type = 2"
     "statement": "test_func(2)"
  },
}


test_ids = []
test_results = []
# for id, library in timeit_dictionary.items():
for _ in range(50):
    id = np.random.randint(0,2)
    library = timeit_dictionary[id]
    result = timeit.repeat(stmt=library["statement"],
        setup=library["setup"],
        number=number_of_times,
        repeat=number_of_repeats)
    test_results.append(np.mean(result))
    test_ids.append(id)

print(np.mean([x for i, x in enumerate(test_results) if test_ids[i]==0]))
print(np.mean([x for i, x in enumerate(test_results) if test_ids[i]==1]))
print()