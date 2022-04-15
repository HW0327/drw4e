<p>
  The package consists of four main modules, which are mixture of Gaussian and Student’s t (Gt), mixture of Gaussian and Gaussian (GG), Gaussian (G) and Student’s t (t). Each module contains eight functions, calculating mu (μ), log 10 of sigma (σ), log 10 of tau (τ), accept rate for tau, degree of freedom, accept rate for degree of freedom, theta (θ) and rate for Z. Since the package returns a list includes all the parameters, if the users want to retrieve any of the parameters, they simply specify the index of it in the list. Here is the index for each parameter.
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
  For example, if you want to the 100 samples values of mu with Gt model, you can type the code below.
</p>

<pre>
import drw4e as dr
gt = dr.Gt(data=your_data, nsample=100, nwarm=100)
mu = gt[0]
Print (mu)
</pre>

<p>
  “your_data” illustrates the timeseries datafile you want to fit, and the required format has been descripted in <a href="https://github.com/HW0327/drw4e/blob/main/data%20descriptions.md">Description of data </a>. Running the code above returns a list of sample mu, with the length of 100, as shown below.
</p>
