HYPRE_INC = -I$(HOME)/install/hypre-2.0.0/src/hypre/include
HYPRE_LIB =  -L$(HOME)/install/hypre-2.0.0/src/hypre/lib -lHYPRE

CXX = g++44
CXXFLAGS = -O3 -g -DHYPRE_SEQUENTIAL -DHYPRE_USING_OPENMP $(HYPRE_INC) -fopenmp
CLIBS =  $(HYPRE_LIB) 

CLAPACK = $(HOME)/install/CLAPACK-3.2.1
INCDIRS = -I$(CLAPACK)/SRC -I$(CLAPACK)
F2CDIR  = $(CLAPACK)/F2CLIBS
LDLIBS  = $(CLAPACK)/lapack_LINUX.a \
          $(CLAPACK)/blas_LINUX.a \
          $(F2CDIR)/libf2c.a -lm 
#$(LDLIBS)

paco: paco.o 
	$(CXX) -o $@ $^ $(CXXFLAGS) $(CLIBS) -L/usr/lib64/ -llapack -lblas -lm -lg2c 

paco.o: paco.cpp paco.hpp quad.hpp
	$(CXX) $(CXXFLAGS) -c paco.cpp 

clean:
	rm *.o