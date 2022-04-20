<p>
  The package requires users to fit a three-column data file into the models. The first column should be the time of each observation in ascending order, the second column is the magnitude of brightness, and the third column is the error for each observation. If your datafile contains additional columns, be sure to filter out these three columns before bringing them into the program. In most cases, each column of data will have a header, so skip those header rows to avoid errors when loading your data. Detailed instructions for loading text files can be found <a href="https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html">here </a>. Below is an example of the datafile.

</p>
<img src="https://github.com/HW0327/drw4e/blob/main/graphs/datasource.png" width="400" height="300"/>
<p>
  These three columns were extracted from macho.dat. The macho.dat is the brightness time series data of MACHO source 70.11469.82, and are irregularly observed via an R-band optical filter on 242 nights for 7.5 years since 1992. 
</p>
