<p>
  Each measurement in this package calculates eight parameters, mu (μ), log 10 of sigma (σ), log 10 of tau (τ), accept rate for tau, degree of freedom, accept rate for degree of freedom, theta (θ) and rate for Z. Since the package returns a list includes all the parameters, if the users want to retrieve any of the parameters, they simply specify the index of it in the list. Here is the index for each parameter.
  </p>

<table>
  <tr>
    <td>mu</td>
    <td>0</td>
  </tr>
  <tr>
    <td>sigma</td>
    <td>1</td>
  </tr>
  <tr>
    <td>tau</td>
    <td>2</td>
  <tr>
    <td>accept rate for tau</td>
    <td>3</td>
  <tr>
    <td>degree of freedom</td>
    <td>4</td>
  <tr>
    <td>accept rate for degree of freedom</td>
    <td>5</td>
  <tr>
    <td>theta</td>
    <td>6</td>
  <tr>
    <td>rate for z</td>
    <td>7</td>
  </tr>
</table>

<p>
  For example, if you want the 100 samples values of mu with Gt model, you can type the code below.
</p>

<pre>
import drw4e as dr
gt = dr.Gt(data=your_data, nsample=100, nwarm=100)
mu = gt[0]
print (mu)
</pre>

<p>
  “your_data” illustrates the timeseries datafile you want to fit, and the required format has been descripted in <a href="https://github.com/HW0327/drw4e/blob/main/data%20descriptions.md">Description of data </a>. Running the code above returns a list of sample mu, with the length of 100, as shown below.
</p>
<img src="https://github.com/HW0327/drw4e/blob/main/graphs/example%20of%20outputs.png" width="400" height="300"/>

<p>
  In most cases, 100 sample data is far from enough. In order to be able to see tens of thousands of data clearly, storing the output data in a txt file is necessary. To continue the above example, you can run the following codes:
</p>

<pre>
import numpy as np
datafile_path = "D:/Gt_mu.txt"
np.savetxt(datafile_path , mu)
</pre>

<p>
  the “datafile_path” in the parenthesis defines where you want to save the txt file on your computer, and “mu” illustrates what you want to save in the txt file. If you want a txt file contains more elements such as, column names and specified format, please visit https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html for more detailed instructions. 
  
  Given the output MCMC samples, you can draw pair-wise graphs to have a better sense of the distribution. Here are the examples of four different models fitted with macho.dat. the description of macho.dat can be found in <a href="https://github.com/HW0327/drw4e/blob/main/data%20descriptions.md">Description of data </a>.
</p>

<div>
   <figure>
      <img src="https://github.com/HW0327/drw4e/blob/main/graphs/G%20pairwise.png" alt="G pairwise" width="300" height="200">
      
   </figure>
   <figure>
      <img src="https://github.com/HW0327/drw4e/blob/main/graphs/GG%20pairwise.png" alt="GG pairwise" width="300" height="200">
   </figure>
</div>
  
<div>
   <figure>
      <img src="https://github.com/HW0327/drw4e/blob/main/graphs/Gt%20pairwise.png" alt="Gt pairwise" width="300" height="200">
      
   </figure>
   <figure>
      <img src="https://github.com/HW0327/drw4e/blob/main/graphs/t%20pairwise.png" alt="t pairwise" width="300" height="200">
      
   </figure>
</div>

