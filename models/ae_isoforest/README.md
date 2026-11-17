# Autoencoder + Isolation Forest

An autoencoder is trained on benign traffic, then an Isolation Forest is fitted on its latent features. The score is `-iso.score_samples(Z)`.
