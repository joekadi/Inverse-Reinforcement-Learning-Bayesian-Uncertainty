import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd(r'/Users/joekadi/Documents/University/5thYear/Thesis/Code/MSci-Project/pyTorch', nargout=0)

ret = eng.triarea(2.0,5.0)   #sending input to the function
print("Triarea : ", ret)


eng.exit()