test:
	(cd pfor1; rm -f test.out; make clean &> /dev/null && make &> /dev/null && ./test.host &> test.out)
	(cd pfor1; diff blessed.in test.out)	
	(cd reduce1; rm -f test.out; make clean &> /dev/null && make &> /dev/null && ./test.host &> test.out)
	(cd reduce1; diff blessed.in test.out)
	(cd sc15a; rm -f test.out; make clean &> /dev/null && make &> /dev/null && ./test.host &> test.out)
	(cd sc15a; diff blessed.in test.out)	
