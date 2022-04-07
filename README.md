# DRW4E:  A Python Package dedicated to estimate brightness variation characteristics of quasars
##Summary
DRW4E represents damped random walk fitted by four error assumptions. 
Two of the error assumptions have been used previously, they are respectively Gaussian error assumption
(Kelly et al. 2009) and mixture of Gaussian and Student's error assumption (Tak et al. 2019), 
while the remaining two have never been used before.
They are Student's error assumption and Mixture of Gaussian and Gaussian error assumption respectively. 
These four assumptions were originally made to estimate and draw the distributions for the brightness 
variations and timescale of massive compact halo objects (MACHO) quasar based on the irregularly observed 
brightness time series data of MACHO via an R-band optical filter on 242 nights for 7.5 years since 1992.
Now they are made into an open-source package for any subsequent similar studies.
##Statement of need
The DRW4E package implements damped random walk algorithm in Python, a high-level, 
general-purpose programming language, intensively used in astrophysics community. 
Potential users could fit their datasets into the package, and the dataset should 
be in the format of a file containing three columns of data, namely observation time, 
brightness and observation errors. Running the program produces a table with eight columns 
of data for the brightness characteristics of the user's research object and parameters involved 
in the calculation. The number of rows depends on the size of Markov chain samples input by the user.
