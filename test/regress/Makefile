ifdef BUILD_NTHREADS
  build_flags := -j$(BUILD_NTHREADS)
endif


test:
	(cd pfor1; rm -f test.out; $(MAKE) clean &> /dev/null && $(MAKE) $(build_flags) &> /dev/null && ./test.host &> test.out)
	(cd pfor1; diff blessed.in test.out)	
	(cd reduce1; rm -f test.out; $(MAKE) clean &> /dev/null && $(MAKE) $(build_flags) &> /dev/null && ./test.host &> test.out)
	(cd reduce1; diff blessed.in test.out)
	(cd sc15a; rm -f test.out; $(MAKE) clean &> /dev/null && $(MAKE) $(build_flags) &> /dev/null && ./test.host &> test.out)
	(cd sc15a; diff blessed.in test.out)	
