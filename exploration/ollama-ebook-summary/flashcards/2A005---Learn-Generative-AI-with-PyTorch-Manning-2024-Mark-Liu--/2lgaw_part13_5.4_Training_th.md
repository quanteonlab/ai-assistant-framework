# Flashcards: 2A005---Learn-Generative-AI-with-PyTorch-Manning-2024-Mark-Liu--_processed (Part 13)

**Starting Chapter:** 5.4 Training the cGAN. 5.4.1 Adding labels to inputs

---

#### Interpolating Real and Fake Images
Background context: In this section, we discuss how to create a continuous blend of real images and generated (fake) images by interpolating between them. This process involves randomly sampling points along a straight line between real and fake images.

:p How do you interpolate between real and fake images?
??x
To create interpolated images, you start with one end being the real image and the other end being the fake image. You then take random samples at various points in between to generate intermediate images that blend characteristics from both ends. This is often visualized as moving a slider between two images.

For example:
- Real Image: \( R \)
- Fake Image: \( F \)

An interpolated image might be represented as \( I = (1 - t)R + tF \), where \( t \in [0, 1] \).

The code for this can look like:

```python
def interpolate_images(real_image, fake_image):
    # Assume real and fake images are tensors
    alpha = torch.rand(1)
    interpolated_image = (1 - alpha) * real_image + alpha * fake_image
    return interpolated_image

real_image = ...  # Load a real image tensor
fake_image = ...  # Load a fake image tensor
interpolated_image = interpolate_images(real_image, fake_image)
```
x??

---

#### Gradient Penalty Calculation
Background context: After presenting the interpolated images to the critic network and obtaining ratings, the next step is to calculate the gradient penalty. This involves measuring how well the gradients of the critic's output are aligned with expectations.

:p What is a gradient penalty in this context?
??x
A gradient penalty is calculated as the squared deviation of the gradient norms from the target value of 1. It measures whether the gradients produced by the interpolations are similar to what they should be, helping ensure that the training process maintains stability and fairness.

The formula for calculating the gradient penalty can be written as:

\[ \text{Gradient Penalty} = \left\| \nabla_{\tilde{x}} f(\tilde{x}) - 1 \right\|^2 \]

Where \( \tilde{x} \) is a randomly chosen point on the line between real and fake images, and \( f(\cdot) \) represents the critic's output.

:p How do you implement gradient penalty in practice?
??x
In practice, after generating interpolated images and getting their scores from the critic network, you compute gradients of these scores with respect to the interpolated images. Then, you calculate the norm of these gradients and penalize them based on how far they are from 1.

```python
import torch

# Assuming we have a batch of interpolated images `interpolated_images`
# and their corresponding critic scores `critic_scores`

# Compute gradients using autograd
gradients = torch.autograd.grad(
    outputs=critic_scores.sum(), 
    inputs=interpolated_images,
    create_graph=True
)[0]

# Reshape the gradients to 1-D vector for easier computation of norms
gradient_norms = gradients.view(gradients.size(0), -1).norm(p=2, dim=1)

# Calculate gradient penalty
gradient_penalty = torch.mean((gradient_norms - 1) ** 2)
```

This code snippet calculates the gradient norm and penalizes it based on its deviation from 1.

x??

---

#### Adding Labels to Inputs for cGAN Training
Background context: In this section, we discuss how to add labels to the inputs of both the generator and critic networks. This is necessary because the goal is to train a conditional GAN (cGAN) that can generate images based on specific characteristics like having or not having eyeglasses.

:p How do you add one-hot labels to input images for cGAN training?
??x
To add one-hot labels to the inputs of both the generator and critic, follow these steps:

1. **Preprocess Data**: Convert images to torch tensors.
2. **Add One-Hot Labels**: Create two one-hot vectors, each representing a class (e.g., with or without eyeglasses).
3. **Concatenate Labels with Input Images**:
   - For the generator: Concatenate random noise vectors and labels before feeding them into the network.
   - For the critic: Attach the one-hot label to the image by adding extra channels.

Here’s how you can do it in code:

```python
import torch

# Example of creating a one-hot vector for an input image with glasses (label 0)
def create_one_hot_label(label):
    one_hot = torch.zeros((2))
    one_hot[label] = 1
    return one_hot

# Convert images to tensors and preprocess them
imgsz = 256
transform = T.Compose([
    T.Resize((imgsz, imgsz)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

data_set = torchvision.datasets.ImageFolder(root="files/glasses", transform=transform)

# Add labels to the images
new_data = []
for i, (img, label) in enumerate(data_set):
    one_hot_label = create_one_hot_label(label)
    
    # Convert image and one-hot label to tensors
    img_tensor = transform(img)
    
    # Attach one-hot label as extra channels
    channels = torch.zeros((2, 256, 256))
    if label == 0:
        channels[0] = 1  # Glasses present: fourth channel filled with 1s
    else:
        channels[1] = 1  # No glasses: fifth channel filled with 1s
    
    img_and_label = torch.cat([img_tensor, channels], dim=0)
    
    new_data.append((img_tensor, label, one_hot_label, img_and_label))
```

x??

---

#### Training cGAN Using Wasserstein Distance
Background context: The next step is to train the conditional GAN (cGAN) using the Wasserstein distance. This approach aims to minimize the Wasserstein distance between the generated image distribution and the real image distribution.

:p How do you train a cGAN with Wasserstein distance?
??x
Training a cGAN with Wasserstein distance involves optimizing both the generator \(G\) and critic \(D\). The objective for the generator is to maximize the expected score from the critic, while the critic aims to minimize the difference between its scores on real and fake images.

The training process can be outlined as follows:

1. **Generate Fake Images**: Sample random noise vectors and pass them through the generator.
2. **Compute Critic Scores**: Pass both real and fake images through the critic network and compute their scores.
3. **Calculate Gradient Penalty**: Compute the gradient penalty to ensure the gradients are well-behaved.
4. **Update Critic**: Update the critic using the loss that minimizes the Wasserstein distance.
5. **Update Generator**: Update the generator to maximize the expected score from the critic.

Here is a high-level pseudocode for training:

```python
def train_gan(generator, discriminator, dataset, num_epochs):
    for epoch in range(num_epochs):
        for real_images, _ in dataset:
            # 1. Train Discriminator
            fake_images = generator(torch.randn(batch_size, noise_dim))
            
            real_scores = discriminator(real_images)
            fake_scores = discriminator(fake_images.detach())
            
            critic_loss_real = -real_scores.mean()
            critic_loss_fake = fake_scores.mean()
            gradient_penalty = compute_gradient_penalty(discriminator, real_images, fake_images)
            total_critic_loss = (critic_loss_real + critic_loss_fake) / 2 + lambda_gp * gradient_penalty
            
            discriminator.zero_grad()
            total_critic_loss.backward()
            optimizer_discriminator.step()
            
            # 2. Train Generator
            fake_scores = discriminator(fake_images)
            generator_loss = -fake_scores.mean()
            
            generator.zero_grad()
            generator_loss.backward()
            optimizer_generator.step()
```

This pseudocode outlines the steps to train a cGAN using Wasserstein distance.

x??

---

#### Background on Image Labeling and Data Preparation

Background context: When loading images using `torchvision.datasets.ImageFolder()` from a folder `/files/glasses`, PyTorch assigns labels to images based on alphabetical order of subfolders. For example, images in `/files/glasses/G/` are labeled as 0 (glasses), and those in `/files/glasses/NoG/` are labeled as 1 (no glasses).

:p Explain the labeling process for images loaded from `/files/glasses`.
??x
Images are labeled based on their subfolder names. For instance, any image under the folder `/files/glasses/G/` will be assigned a label of 0 indicating it has glasses, and an image in `/files/glasses/NoG/` is labeled as 1, indicating it does not have glasses.
x??

---

#### Creating Data with Labels for Training

Background context: To train a Conditional GAN (cGAN), you need to create images with specific characteristics such as having or not having glasses. You can achieve this by modifying the input vector fed into the generator.

:p How do you modify the input vector to indicate that an image should have glasses?
??x
You add a one-hot label [1, 0] to the noise vector before feeding it into the generator. This tells the generator that the output image should have glasses.
x??

---

#### Training the Critic and Generator

Background context: The training process involves alternating between training the critic and the generator. The critic is trained on both real and fake images, while the generator is trained to produce realistic-looking images.

:p What are the three components of the loss function for the critic in the `train_batch()` function?
??x
The loss function for the critic has three components:
1. Loss from evaluating real images.
2. Loss from evaluating fake images.
3. Gradient penalty loss.

This is represented by the formula:
\[ \text{loss\_critic} = -(\text{torch.mean(critic\_real) - torch.mean(critic\_fake)}) + 10 * gp \]
x??

---

#### Training the Generator

Background context: The generator is trained to produce images that the critic cannot distinguish from real images. This involves minimizing a loss function derived from the critic's evaluation of fake images.

:p How does the `train_batch()` function train the generator?
??x
The `train_batch()` function trains the generator by using the Wasserstein loss, which aims to minimize the difference between the critic's evaluations of real and fake images. Specifically:
1. It first generates a batch of fake images using random noise concatenated with labels.
2. The generator is then trained on these fake images.

This is implemented in the code as follows:
```python
gen_fake = critic(fake).reshape(-1)
loss_gen = -torch.mean(gen_fake)
gen.zero_grad()
loss_gen.backward()
opt_gen.step()
```
x??

---

#### Plotting Generated Images

Background context: Periodically inspecting generated images helps assess the performance of the model. This involves plotting images with and without glasses to visually evaluate their quality.

:p How does the `plot_epoch()` function create and plot images with glasses?
??x
The `plot_epoch()` function creates one-hot labels [1, 0] for images that should have glasses and uses these labels along with random noise vectors. It then feeds these concatenated vectors into the generator to produce images with glasses.

Here's a simplified version of how it works:
```python
noise = torch.randn(32, z_dim, 1, 1)
labels = torch.zeros(32, 2, 1, 1)
labels[:,0,:,:]=1         # Create one-hot labels for images with glasses

noise_and_labels=torch.cat([noise,labels],dim=1).to(device)
fake=gen(noise_and_labels).cpu().detach()

# Plot the generated images
fig=plt.figure(figsize=(20,10),dpi=100)
for i in range(32):
    ax = plt.subplot(4, 8, i + 1)
    img=(fake.cpu().detach()[i]/2+0.5).permute(1,2,0)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

plt.subplots_adjust(hspace=-0.6)
plt.savefig(f"files/glasses/G{epoch}.png")
plt.show()
```
x??

---

#### Training the cGAN

Background context: The Conditional GAN (cGAN) is trained using batches of data, alternating between training the critic and generator. This process helps improve image quality by ensuring that both components work effectively together.

:p How does the `train_batch()` function handle the training steps for both the critic and generator?
??x
The `train_batch()` function handles the training in two main parts:
1. Training the critic: It evaluates real and fake images, calculates the loss including gradient penalty, and updates the critic's parameters.
2. Training the generator: It uses the output of the critic on fake images to update the generator's parameters.

Here is a simplified version of the `train_batch()` function logic:
```python
def train_batch(onehots, img_and_labels, epoch):
    real = img_and_labels.to(device)
    B = real.shape[0]
    for _ in range(5):
        noise = torch.randn(B, z_dim, 1, 1)
        onehots=onehots.reshape(B,2,1,1)
        noise_and_labels=torch.cat([noise,onehots],dim=1).to(device)
        fake_img = gen(noise_and_labels).to(device)
        fakelabels=img_and_labels[:,3:,:,:].to(device)
        fake=torch.cat([fake_img,fakelabels],dim=1).to(device)

        critic_real = critic(real).reshape(-1)
        critic_fake = critic(fake).reshape(-1)
        gp = GP(critic, real, fake)

        loss_critic=(-(torch.mean(critic_real) - torch.mean(critic_fake)) + 10 * gp)
        critic.zero_grad()
        loss_critic.backward(retain_graph=True)
        opt_critic.step()

    gen_fake = critic(fake).reshape(-1)
    loss_gen = -torch.mean(gen_fake)
    gen.zero_grad()
    loss_gen.backward()
    opt_gen.step()

    return loss_critic, loss_gen
```
x??

---

#### Training Loop for Conditional GANs
Background context: The training loop involves iterating through batches of data, training the generator and critic models, and calculating loss values. This process helps in refining the model to generate images with specific characteristics based on input labels.

:p What does the provided code snippet show about the training process?
??x
The provided code snippet demonstrates how to train a Conditional Generative Adversarial Network (cGAN) for generating images. It iterates over all batches of data, trains both the critic and generator models, and calculates their respective losses. After each epoch, it prints out the critic and generator loss values.

```python
for _,_,onehots,img_and_labels in data_loader:
    loss_critic, loss_gen = train_batch(onehots,\                                 img_and_labels,epoch)
    closs += loss_critic.detach() / len(data_loader)
gloss += loss_gen.detach() / len(data_loader)

print(f"at epoch {epoch},\     critic loss: {closs}, generator loss {gloss}")
plot_epoch(epoch) 
torch.save(gen.state_dict(), 'files/cgan.pth')
```

- `train_batch(onehots, img_and_labels, epoch)`: This function trains the model on a batch of data. It returns the losses for both the critic and the generator.
- `closs` and `gloss`: These are accumulators that keep track of the total loss for each epoch to compute the average over all batches.

This training loop continues until the desired number of epochs is completed, and at the end, it saves the trained model's weights.
x??

---

#### Generating Images with Specific Characteristics
Background context: The text describes two methods to generate images with specific characteristics—by attaching a label to random noise or by selecting specific noise vectors. The first method involves using cGANs, while the second uses handpicked noise vectors.

:p How does the code snippet demonstrate generating images with eyeglasses?
??x
The provided code snippet demonstrates how to use a trained conditional GAN (cGAN) model to generate 32 images of human faces with eyeglasses. The process involves setting up the generator, loading its weights, and feeding it with a combination of random noise vectors and labels.

```python
torch.manual_seed(0)
generator = Generator(z_dim+2, img_channels, features).to(device)
generator.load_state_dict(torch.load("files/cgan.pth", map_location=device))
generator.eval()

noise_g = torch.randn(32, z_dim, 1, 1)
labels_g = torch.zeros(32, 2, 1, 1)
labels_g[:,0,:,:] = 1

noise_and_labels = torch.cat([noise_g, labels_g], dim=1).to(device)
fake = generator(noise_and_labels)

plt.figure(figsize=(20, 10), dpi=50)
for i in range(32):
    ax = plt.subplot(4, 8, i + 1)
    img = (fake.cpu().detach()[i] / 2 + 0.5).permute(1, 2, 0)
    plt.imshow(img.numpy())
    plt.xticks([])
    plt.yticks([])

plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
plt.show()
```

- `torch.manual_seed(0)`: This ensures that the random numbers generated are reproducible.
- `generator = Generator(z_dim+2, img_channels, features).to(device)`: Creates an instance of the generator model and moves it to the specified device (CPU or GPU).
- `noise_g` and `labels_g`: These variables represent 32 random noise vectors and corresponding labels indicating that the images should have eyeglasses.
- `torch.cat([noise_g, labels_g], dim=1)`: Combines the noise vectors and labels into a single tensor to be fed into the generator.

This process generates 32 images with eyeglasses as requested.
x??

---

#### Selecting Characteristics by Label Interpolation
Background context: The text explains how label interpolation can be used in cGANs to generate intermediate types of images. This involves creating weighted averages of labels to produce images that have a mix of the characteristic represented by each label.

:p How is label interpolation used to perform label arithmetic in cGANs?
??x
Label interpolation in Conditional GANs (cGANs) involves using interpolated labels between two distinct states to generate intermediate types of images. For example, if we want an image that has a mix of both eyeglasses and no-eyeglasses characteristics, we can interpolate the labels [1,0] and [0,1].

```python
weights = [0, 0.25, 0.5, 0.75, 1]
plt.figure(figsize=(20,4), dpi=300)
for i in range(5):
    ax = plt.subplot(1, 5, i + 1)

    # Change the value of z
    label = weights[i] * labels_ng[0] + (1 - weights[i]) * labels_g[0]

    noise_and_labels = torch.cat(
        [z_female_g.reshape(1, z_dim, 1, 1), 
         label.reshape(1, 2, 1, 1)], dim=1).to(device)
    
    fake = generator(noise_and_labels).cpu().detach()
    img = (fake[0] / 2 + 0.5).permute(1, 2, 0)

    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
plt.show()
```

- `weights = [0, 0.25, 0.5, 0.75, 1]`: These are the weights used to interpolate between the no eyeglasses and eyeglasses labels.
- `label = weights[i] * labels_ng[0] + (1 - weights[i]) * labels_g[0]`: This line creates a weighted average of the two labels, resulting in an interpolated label.
- The process involves feeding this interpolated label to the generator along with the corresponding noise vector.

This approach allows generating images that exhibit characteristics between the two states defined by the labels. By varying the weights, you can generate different levels of intermediate characteristics.
x??

---

