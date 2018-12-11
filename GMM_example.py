import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
from scipy.stats import entropy as KL


# generate samples from misture of gaussians
def generate_samples(weights, means, cors, num):
    n = len(means)
    ans = []
    for _ in range(num):
        r = np.random.rand()
        ind = 0
        for i in range(n):
            if r <= sum(weights[:(i+1)]):
                ind = i
                break
        ans.append(np.random.normal(means[ind], cors[ind]))
    return np.array(ans)


def gmm(samples, n_clusters):
    model = mixture.GaussianMixture(n_components=n_clusters)
    model.fit([[x] for x in samples])
    return [model.weights_,
            np.array([x[0] for x in model.means_]),
            np.array([x[0][0] for x in model.covariances_])]

# print(gmm(generate_samples([0.3,0.7], [-1,1], [0.2,0.5], 10), 3))


def gaussian_distr(x, mu, rou):
    return np.exp(-(x - mu) ** 2 / (2 * rou ** 2)) / (np.sqrt(2*np.pi) * rou)


def pdf(weights, means, cors, num=100, r=2.0):
    ans = np.zeros(num+1)
    for i,x in enumerate(np.linspace(-r,r,num+1)):
        ans[i] = sum([weights[j] * gaussian_distr(x, means[j], cors[j]) for j in range(len(means))])
    return ans





real_samples = generate_samples([0.3,0.2,0.2,0.3], [-2,0,0.5,2], [0.4,0.4,0.4,0.4], 100)
dist = [[] for _ in range(10)]
for model_clusters in range(1, 10):
    print('----------------------')
    print(model_clusters)
    params = gmm(real_samples, model_clusters)  # generative model G with different generalization properties
    print(params)
    model_samples = generate_samples(params[0], params[1], params[2], 1000)  # sample from G
    for scale in range(1, 10)[::-1]:
        scale_params_sampled = gmm(model_samples, scale)  # samples from G GMM
        scale_params_real = gmm(real_samples, scale)  # real samples GMM
        pdf_sampled = pdf(*scale_params_sampled, r=3)
        pdf_real = pdf(*scale_params_real, r=3)
        #print(pdf_real)
        dist[model_clusters].append(KL(pdf_sampled, pdf_real))
        print(KL(pdf_sampled, pdf_real))


print('----------------------')
plt.subplot(121)
for i in range(1, 10):
    plt.plot([j for j in range(10)], [i]+dist[i], marker = '.')
    print(i, dist[i][0])

plt.subplot(122)
plt.scatter(real_samples, np.zeros(len(real_samples)))

plt.show()