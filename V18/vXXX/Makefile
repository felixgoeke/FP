all: build/vXXX.pdf

# hier Python-Skripte:
build/build/Geschwindigkeit15.pdf build/Geschwindigkeit30.pdf build/Geschwindigkeit60.pdf build/Streui.pdf build/Stroev.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python kalibrierung2.py 
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python Linienbreite_Neu.py 
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python VollEnergie.py
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python Caesium.py
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python Barium.py
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python Uran.py

# hier weitere Abhängigkeiten für build/vXXX.pdf deklarieren:
#build/vXXX.pdf:


build/vXXX.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	vXXX.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
