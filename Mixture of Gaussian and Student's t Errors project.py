## most updated 3/10, all graphs have been figured out

import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import math
import seaborn as sns
import matplotlib.pyplot as plt

def post_OU(time, X, previous_theta, tau_jump, tau_thresh,
            tau_prior_a, tau_prior_b, sigma_prior_a, sigma_prior_b):
    leng_time = len(time)
    mu = previous_theta[0]
    sigma = previous_theta[1]
    tau = previous_theta[2]

    time_diff = np.diff(time)
    a_i = np.exp(- time_diff / tau)  # i = 2, 3, ..., n
    leng_a = len(a_i)  # n - 1

    # updating mu
    mu_mean = (X[0] + sum((X[1:] - a_i * X[:leng_time-1]) / (1 + a_i))) / (1 + sum((1 - a_i) / (1 + a_i)))
    mu_sd = np.sqrt(tau * sigma ** 2 / 2 /
               (1 + sum((1 - a_i) / (1 + a_i))))
    inv_cdf = np.random.uniform(low=sp.stats.norm.cdf(-30, loc=mu_mean, scale=mu_sd),
                  high=sp.stats.norm.cdf(30, loc=mu_mean, scale=mu_sd), size=1)
    mu = sp.stats.norm.ppf(inv_cdf, loc=mu_mean, scale=mu_sd)

    # updating sigma
    sigma = np.sqrt((sigma_prior_b + (X[0] - mu) ** 2 / tau +
                    sum((X[1:] - a_i * X[:leng_time-1] - mu * (1 - a_i)) ** 2 /
                        (1 - a_i ** 2)) / tau) /
                   np.random.gamma(shape=leng_time / 2 + sigma_prior_a, scale=1, size=1))

    # updating tau
    def log_post_tau(t):
        a_i_post = np.exp(- time_diff / t)
        return - (leng_time / 2 + 1 + tau_prior_a) * np.log(t) - 0.5 * sum(np.log(1 - a_i_post ** 2)) - \
               (tau_prior_b + (X[1] - mu) ** 2 / sigma ** 2 + sum((X[1:] - a_i_post * X[:leng_time-1] -
               mu * (1 - a_i_post)) ** 2 / (1 - a_i_post ** 2)) / sigma ** 2) / t

    tau_p = np.exp(np.log(tau) + tau_jump)
    l_metrop = log_post_tau(tau_p) - log_post_tau(tau)
    l_hastings = np.log(tau_p) - np.log(tau)

    # Accept-reject
    if l_metrop + l_hastings > tau_thresh:
        tau = tau_p

    out = list((mu, sigma, tau))
    return out

def post_X(time, x, se_x, X, z, alpha, OU):
    leng_time = len(time)

    lcA = x
    se_lcA = se_x  # 2nd column of the dat

    mu =OU[0]
    sigma = OU[1]
    tau = OU[2]
    # a.i, i = 1, ..., n
    time_diff = np.diff(time)
    a_i = np.exp(-time_diff / tau)

    # x.i, i = 1, 2, ..., n
    X = X - mu
    x = lcA - mu

    # B, i = 1, 2, ..., n to be saved
    B = np.repeat(0.01, leng_time)

    # mu.i, i = 1, 2, ..., 2n to be saved
    mu_i = np.repeat(0.01, leng_time)

    # shrinkages
    var0 = tau * sigma ** 2 / 2
    B[0] = se_lcA[0] ** 2 / (se_lcA[0] ** 2 + var0 * (1 - a_i[0] ** 2))
    mu_i[0] = (1 - B[0]) * x[0] + B[0] * a_i[0] * X[1]
    X[0] = np.random.normal(loc=mu_i[0], scale=np.sqrt(se_lcA[0] ** 2 * (1 - B[0])), size=1)

    for k in range(2, leng_time):
        B[k-1] = se_lcA[k-1] ** 2 / (se_lcA[k-1] ** 2 + var0 * (1 - a_i[k - 2] ** 2) *
                                (1 - a_i[k-1] ** 2) / (1 - (a_i[k - 2] * a_i[k-1]) ** 2))
        mu_i[k-1] = (1 - B[k-1]) * x[k-1] + B[k-1] * (a_i[k-1] * (1 - a_i[k - 2] ** 2) * X[k] +
                                              a_i[k - 2] * (1 - a_i[k-1] ** 2) * X[k - 2]) / \
                  (1 - (a_i[k - 2] * a_i[k-1]) ** 2)
        X[k-1] = np.random.normal(loc=mu_i[k-1], scale=np.sqrt(se_lcA[k-1] ** 2 * (1 - B[k-1])), size=1)

    B[leng_time-1] = se_lcA[leng_time-1] ** 2 / (se_lcA[leng_time-1] ** 2 + var0 * (1 - a_i[leng_time - 2] ** 2))
    mu_i[leng_time-1] = (1 - B[leng_time-1]) * x[leng_time-1] + \
                      B[leng_time-1] * a_i[leng_time - 2] * X[leng_time - 2]
    X[leng_time-1] = np.random.normal(loc=mu_i[leng_time-1],
                                    scale=np.sqrt(se_lcA[leng_time-1] ** 2 * (1 - B[leng_time-1])), size=1)

    return X + mu

def outlier_mixture(data, OU_ini, z_ini, alpha_ini, theta_ini, tau_prior_shape, tau_prior_scale,
                    tau_log_jump_scale, sigma_prior_shape, sigma_prior_scale,
                    a_beta, b_beta, df_scale,
                    t_df, theta_update=False, z_update=False, df_random=False,
                    alpha_update=False,
                    sample_size=50, warmingup_size=50):
    data = np.array(data)
    total_sample_size = sample_size + warmingup_size
    time = data[:, 0]
    leng_time = len(time)

    x = data[:, 1]
    se_x = se_x_original = data[:, 2]

    mu_t = OU_ini[0]  # mu ini, 18.2
    sigma_t = OU_ini[1]  # sigma ini, 0.01
    tau_t = OU_ini[2]  # tau ini, 200
    X_t = x  # 2nd column of the dat
    z_t = z_ini  # np.repeat(0, len(time))
    theta_t = theta_ini  # 0.01
    alpha_t = alpha_ini  # np.repeat(1, len(time))
    df_t = t_df  # 4

    mu_out = np.repeat(0.001, total_sample_size)
    sigma_out = np.repeat(0.001, total_sample_size)
    tau_out = np.repeat(0.001, total_sample_size)
    df_out = np.repeat(0.001, total_sample_size)
    theta_out = np.repeat(0.001, total_sample_size)
    z_out = np.full((total_sample_size, leng_time), 0.001)

    df_accept = np.repeat(0.001, total_sample_size)
    tau_accept = np.repeat(0.001, total_sample_size)
    tau_jumps = tau_log_jump_scale * np.random.normal(size=total_sample_size)  # 1d list, len=100
    tau_thresh = -np.random.exponential(size=total_sample_size)  # 1d list, len=100, all negative
    tau_scale_adapt = 1

    def df_log_cond_post(df):
        return -(df / 2) * sum(np.log(alpha_t) + 1 / alpha_t) + \
               leng_time * (df / 2) * np.log(df / 2) - leng_time * np.log(math.gamma(df / 2))

    for i in range(1, total_sample_size+1):
        # X update
        X_t = post_X(time=time, x=x, se_x=se_x, X=X_t, z=z_t,
                     alpha=alpha_t, OU=list((mu_t, sigma_t, tau_t)))

        # OU update
        tau_jump_adapt = tau_jumps[i-1] * tau_scale_adapt
        OU_update = post_OU(time=time, X=X_t,
                            previous_theta=list((mu_t, sigma_t, tau_t)),
                            tau_jump=tau_jump_adapt, tau_thresh=tau_thresh[i-1],
                            tau_prior_a=tau_prior_shape,
                            tau_prior_b=tau_prior_scale,
                            sigma_prior_a=sigma_prior_shape,
                            sigma_prior_b=sigma_prior_scale)
        if OU_update[2] != tau_t:
            tau_accept[i-1] = 1

        mu_t = mu_out[i - 1]= OU_update[0]
        sigma_t = sigma_out[i - 1] = OU_update[1]
        tau_t = tau_out[i - 1] =  OU_update[2]

        # update theta and z
        if theta_update == True:
            theta_t = float(np.random.beta(a=sum(z_t) + a_beta, b=leng_time - sum(z_t) + b_beta, size=1))
            theta_out[i-1] = theta_t

        if z_update == True:
            q = theta_t * sp.stats.norm.pdf(x=x, loc=X_t, scale=np.sqrt(alpha_t) * se_x_original) /\
                (theta_t * sp.stats.norm.pdf(x=x, loc=X_t, scale=np.sqrt(alpha_t) * se_x_original) +
                (1 - theta_t) * sp.stats.norm.pdf(x=x, loc=X_t, scale=se_x_original))
            z_t = np.random.binomial(size=leng_time, n=1, p=q)
            z_out[i-1] = z_t

        if alpha_update == True:
            alpha_t = ((x - X_t) ** 2 * z_t / se_x_original ** 2 + df_t) / 2 / \
                      np.random.gamma(shape=(z_t + df_t) / 2, size=leng_time)

        if df_random == True:
            df_p = 50
            while df_p > 40 or df_p < 0:
                df_p = np.exp(np.log(df_t) + np.random.normal(scale=df_scale, size=1))

            df_p_den = df_log_cond_post(df_p)
            df_t_den = df_log_cond_post(df_t)
            l_metrop = df_p_den - df_t_den
            l_hastings = np.log(df_p) - np.log(df_t)

            # Accept-reject
            if l_metrop + l_hastings > -np.random.exponential(1):
                df_t = df_p
                df_accept[i-1] = 1

            df_out[i-1] = df_t

            if i % 100 == 0:
                if np.mean(df_accept[i - 100: i]) > 0.35:
                    scale_adj = np.exp(min(0.01, 1 / np.sqrt(i / 100)))
                elif np.mean(df_accept[i - 100: i]) < 0.35:
                    scale_adj = np.exp(-min(0.01, 1 / np.sqrt(i / 100)))
                else:
                    scale_adj = 1
                df_scale = df_scale * scale_adj

        se_x = alpha_t ** (z_t / 2) * se_x_original

        if i % 100 == 0:
            if np.mean(tau_accept[i - 100: i]) > 0.35:
                scale_adj = np.exp(min(0.01, 1 / np.sqrt(i / 100)))
            elif np.mean(tau_accept[i - 100: i]) < 0.35:
                scale_adj = np.exp(-min(0.01, 1 / np.sqrt(i / 100)))
            else:
                scale_adj = 1
            tau_scale = tau_scale_adapt * scale_adj

    mu = mu_out[warmingup_size:],
    sigma = sigma_out[warmingup_size:],
    tau = tau_out[warmingup_size:],
    tau_accept_rate = np.mean(tau_accept),
    df = df_out[warmingup_size:],  # id array, last 50 elements of np.repeat('NA', 100)
    df_accept_rate = np.mean(df_accept),
    theta = theta_out[warmingup_size:],
    z_out = pd.DataFrame(z_out[warmingup_size:, ])
    z_rate = z_out.mean()
    out = list((mu, sigma, tau, tau_accept_rate, df, df_accept_rate, theta, z_rate))
    return out

def mode_est(s):
    d = sns.displot(s)
    # return d(x)[which.max(d(y))] ???

################# MACHO light curves

dat_temp = np.loadtxt("macho.dat", skiprows=2)

# removing the data with negative measurement errors

dat = dat_temp[:, [0, 3, 4]]
dat = dat[dat[:,2]>0, :]

time = dat[:, 0]

n_sample = 1000
n_warm = 1000

res_Gt = outlier_mixture(data=dat, z_ini=np.repeat(0, len(time)), theta_ini=0.01,
                         alpha_ini=np.repeat(1, len(time)), OU_ini=list((18.2, 0.01, 200)),
                         tau_prior_shape=1, tau_prior_scale=1,
                         tau_log_jump_scale=1.5, z_update=True,
                         a_beta=2.42, b_beta=239.58,
                         alpha_update=True, df_scale=0.23,
                         sigma_prior_shape=1, sigma_prior_scale=10 ** (-7),
                         t_df=4, theta_update=True, df_random=True,
                         sample_size=n_sample, warmingup_size=n_warm)

res_GG = outlier_mixture(data = dat, z_ini=np.repeat(0, len(time)), theta_ini=0.01,
                         alpha_ini=np.repeat(1e2, len(time)), OU_ini=list((18.2, 0.01, 200)),
                         tau_prior_shape=1, tau_prior_scale=1,
                         tau_log_jump_scale=1.5, z_update=True, df_scale=0.23,
                         a_beta=2.42, b_beta=239.58,
                         sigma_prior_shape=1, sigma_prior_scale=10 ** (-7),
                         t_df=4, theta_update=True, df_random=False,
                         alpha_update=False, sample_size=n_sample, warmingup_size=n_warm)

res_G = outlier_mixture(data=dat, z_ini=np.repeat(0, len(time)), theta_ini=np.repeat(0, len(time)),
                        alpha_ini=np.repeat(1, len(time)), OU_ini=list((18.2, 0.01, 200)),
                        tau_prior_shape=1, tau_prior_scale=1,
                        tau_log_jump_scale=0.75, z_update=False, df_scale=0.23, a_beta=2.42, b_beta=239.58,
                        sigma_prior_shape=1, sigma_prior_scale=10 ** (-7),
                        t_df=4, theta_update=False, sample_size=n_sample, warmingup_size=n_warm)

res_t = outlier_mixture(data=dat, z_ini=np.repeat(1, len(time)), theta_ini=np.repeat(1, len(time)),
                        alpha_ini=np.repeat(1, len(time)), OU_ini=list((18.2, 0.01, 200)),
                        a_beta=2.42, b_beta=239.58, alpha_update=True, df_scale=0.26,
                        tau_prior_shape=1, tau_prior_scale=1,
                        tau_log_jump_scale=1.75, z_update=False,
                        sigma_prior_shape=1, sigma_prior_scale=10 ** (-7),
                        t_df=4, theta_update=False, df_random=True,
                        sample_size=n_sample, warmingup_size=n_warm)

fgtmu = res_Gt[0]
fgtsigma = np.log(res_Gt[1])
fgtsigma = fgtsigma.flatten()
fgttau = np.log(res_Gt[2])
fgttau = fgttau.flatten()

fggmu = res_GG[0]
fggsigma = np.log(res_GG[1])
fggsigma = fggsigma.flatten()
fggtau = np.log(res_GG[2])
fggtau = fggtau.flatten()

ftmu = res_t[0]
ftsigma = np.log(res_t[1])
ftsigma = ftsigma.flatten()
fttau = np.log(res_t[2])
fttau = fttau.flatten()

fgmu = res_G[0]
fgsigma = np.log(res_G[1])
fgsigma = fgsigma.flatten()
fgtau = np.log(res_G[2])
fgtau = fgtau.flatten()

fig = plt.figure()
ax1 = fig.add_subplot(3, 4, 1)
ax2 = fig.add_subplot(3, 4, 2)
ax3 = fig.add_subplot(3, 4, 3)
ax4 = fig.add_subplot(3, 4, 4)
ax5 = fig.add_subplot(3, 4, 5)
ax6 = fig.add_subplot(3, 4, 6)
ax7 = fig.add_subplot(3, 4, 7)
ax8 = fig.add_subplot(3, 4, 8)
ax9 = fig.add_subplot(3, 4, 9)
ax10 = fig.add_subplot(3, 4, 10)
ax11 = fig.add_subplot(3, 4, 11)
ax12 = fig.add_subplot(3, 4, 12)
fig.tight_layout(pad=3.0)

n, bins, patches = ax1.hist(ftmu, alpha=0.5, histtype='bar', ec='black')
ax1.set_xlabel('res.t(mu)')
ax1.set_ylabel('Frequency')

n, bins, patches = ax2.hist(fgmu, alpha=0.5, histtype='bar', ec='black')
ax2.set_xlabel('res.G(mu)')
ax2.set_ylabel('Frequency')

n, bins, patches = ax3.hist(fgtmu, alpha=0.5, histtype='bar', ec='black')
ax3.set_xlabel('res.Gt(mu)')
ax3.set_ylabel('Frequency')

n, bins, patches = ax4.hist(fggmu, alpha=0.5, histtype='bar', ec='black')
ax4.set_xlabel('res.GG(mu)')
ax4.set_ylabel('Frequency')

n, bins, patches = ax5.hist(fttau, alpha=0.5, histtype='bar', ec='black')
ax5.set_xlabel('log(res.t(tau))')
ax5.set_ylabel('Frequency')

n, bins, patches = ax6.hist(fgtau, alpha=0.5, histtype='bar', ec='black')
ax6.set_xlabel('log(res.G(tau))')
ax6.set_ylabel('Frequency')

n, bins, patches = ax7.hist(fgttau, alpha=0.5, histtype='bar', ec='black')
ax7.set_xlabel('log(res.Gt(tau))')
ax7.set_ylabel('Frequency')

n, bins, patches = ax8.hist(fggtau, alpha=0.5, histtype='bar', ec='black')
ax8.set_xlabel('log(res.GG(tau))')
ax8.set_ylabel('Frequency')

n, bins, patches = ax9.hist(ftsigma, alpha=0.5, histtype='bar', ec='black')
ax9.set_xlabel('log(res.t(sigma))')
ax9.set_ylabel('Frequency')

n, bins, patches = ax10.hist(fgsigma, alpha=0.5, histtype='bar', ec='black')
ax10.set_xlabel('log(res.G(sigma))')
ax10.set_ylabel('Frequency')

n, bins, patches = ax11.hist(fgtsigma, alpha=0.5, histtype='bar', ec='black')
ax11.set_xlabel('log(res.Gt(sigma))')
ax11.set_ylabel('Frequency')

n, bins, patches = ax12.hist(fggsigma, alpha=0.5, histtype='bar', ec='black')
ax12.set_xlabel('log(res.GG(sigma))')
ax12.set_ylabel('Frequency')

plt.show()
