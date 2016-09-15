CXX = nvcc
CFLAGS = -g 

Target =ysu_bd_gpu.cu

$(basename $(Target)) : $(Target)
	$(CXX) $(CFLAGS) -o $@ $<

.PHONY : memoryleak_check

memoryleak_check:$(basename $(Target))
	valgrind --tool=memcheck --leak-check=yes --track-origins=yes -v ./$(basename $(Target))

.PHONY : time_pro
time_pro : $(basename $(Target))
	./$(basename $(Target))
	gprof $(basename $(Target)) gmon.out > $(basename $(Target)).txt

.PHONY : compact
compact:
	rm -rf *.out *.vtk

.PHONY : clean
clean:
	rm -rf *.dat *.out *.hist *.o *.xyz *~ *log *err $(basename $(Target)) *.vtk
