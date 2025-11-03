# Flashcards: 2A005---Learn-Generative-AI-with-PyTorch-Manning-2024-Mark-Liu--_processed (Part 14)

**Starting Chapter:** 5.5.2 Vector arithmetic in latent space

---

#### Label Arithmetic in cGAN
Label arithmetic allows us to explore the feature space of generated images by interpolating between different labels. In this context, we are using a conditional GAN (cGAN) where the generator is conditioned on a label that specifies attributes such as whether an image should have eyeglasses or not.
:p What does label arithmetic in cGAN allow us to do?
??x
Label arithmetic allows us to interpolate between different labels to explore the feature space of generated images. By using weighted averages of two labels, we can generate images with varying degrees of the specified attribute. For example, we can create images that gradually transition from having eyeglasses to not having them.
```python
# Example code for label arithmetic
labels_ng = torch.tensor([[0]])  # No eyeglasses label
labels_g = torch.tensor([[1]])   # Eyeglasses label

weights = [0, 0.25, 0.5, 0.75, 1]

plt.figure(figsize=(20,4), dpi=50)
for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    
    # Create a weighted average of the two labels
    label = weights[i] * labels_ng + (1 - weights[i]) * labels_g
    
    noise_and_label = torch.cat([z_female_g.reshape(1, z_dim, 1, 1), 
                                 label.reshape(1, 2, 1, 1)], dim=1).to(device)
    
    fake = generator(noise_and_label).cpu().detach()
    img = (fake[0]/2+0.5).permute(1,2,0)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
plt.show()
```
x??

---

#### Vector Arithmetic in Latent Space
Vector arithmetic allows us to manipulate the characteristics of generated images by interpolating between different latent vectors (noise). This technique can be used to select and combine features such as gender.
:p How does vector arithmetic work in the context of generating human faces?
??x
Vector arithmetic involves creating a weighted average of two or more random noise vectors in the latent space. By doing so, we can generate images that have a blend of the characteristics associated with those noise vectors. For example, we can create images that show a transition from male to female features.
```python
# Example code for vector arithmetic
weights = [0, 0.25, 0.5, 0.75, 1]

plt.figure(figsize=(20,4), dpi=50)
for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    
    # Create a weighted average of the two noise vectors
    z = weights[i] * z_female_ng + (1 - weights[i]) * z_male_ng
    
    noise_and_label = torch.cat([z.reshape(1, z_dim, 1, 1), 
                                 labels_ng[0].reshape(1, 2, 1, 1)], dim=1).to(device)
    
    fake = generator(noise_and_label).cpu().detach()
    img = (fake[0]/2+0.5).permute(1,2,0)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
plt.show()
```
x??

---

#### Changing Noise Vectors in Generated Images
By changing the noise vector used during image generation, we can influence the characteristics of the generated images. For instance, using a noise vector that corresponds to male features will generate an image with a male face.
:p How does changing the noise vector affect the generated images?
??x
Changing the noise vector affects the characteristics of the generated images by altering the input to the generator model. Different noise vectors can represent different types of faces (e.g., male vs. female). By substituting `z_female_g` with `z_male_g`, we generate images that have a male face instead of a female one.
```python
# Example code for changing noise vector
weights = [0, 0.25, 0.5, 0.75, 1]

plt.figure(figsize=(20,4), dpi=50)
for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    
    # Change the noise vector
    z = weights[i] * z_female_g + (1 - weights[i]) * z_male_g
    
    noise_and_label = torch.cat([z.reshape(1, z_dim, 1, 1), 
                                 labels_ng[0].reshape(1, 2, 1, 1)], dim=1).to(device)
    
    fake = generator(noise_and_label).cpu().detach()
    img = (fake[0]/2+0.5).permute(1,2,0)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
plt.show()
```
x??

---

#### Changing Labels in Generated Images
Changing the label used during image generation can alter the attribute of the generated images. For example, changing from a no-eyeglasses label to an eyeglasses label will generate images that have or do not have eyeglasses.
:p How does changing the label affect the generated images?
??x
Changing the label affects the attribute of the generated images by providing different conditions to the generator model. Labels instruct the generator on what features to include in the image. By switching from `labels_ng` (no eyeglasses) to `labels_g` (eyeglasses), we generate images with or without eyeglasses.
```python
# Example code for changing labels
weights = [0, 0.25, 0.5, 0.75, 1]

plt.figure(figsize=(20,4), dpi=50)
for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    
    # Change the label
    label = weights[i] * labels_ng + (1 - weights[i]) * labels_g
    
    noise_and_label = torch.cat([z_female_g.reshape(1, z_dim, 1, 1), 
                                 label.reshape(1, 2, 1, 1)], dim=1).to(device)
    
    fake = generator(noise_and_label).cpu().detach()
    img = (fake[0]/2+0.5).permute(1,2,0)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
plt.show()
```
x??

#### Selecting Two Characteristics Simultaneously
Background context explaining how selecting one characteristic at a time is different from selecting two characteristics simultaneously. Mention that so far, you have learned to generate images based on single characteristics like eyeglasses or gender using specific noise vectors and labels. The goal now is to combine these into selecting both eyeglasses and gender in the same image.

:p How do we select two characteristics (like glasses and gender) simultaneously?
??x
To select two characteristics simultaneously, such as glasses and gender, you can use a combination of a random noise vector and a label. This involves generating images with both eyeglasses and different genders by iterating through four possible cases: male faces with glasses, male faces without glasses, female faces with glasses, and female faces without glasses.

Here's the detailed process:

1. **Define Noise Vectors**: You have two noise vectors `z_female_g` (noise vector for generating female faces with glasses) and `z_male_g` (noise vector for generating male faces with glasses).
2. **Define Labels**: You also need to define labels where `labels_ng[0]` is a label for no glasses, and `labels_g[0]` is a label for glasses.
3. **Iterate Through Cases**: Use four iterations (`i = 0` to `3`) to generate images based on the combinations of gender (male or female) and presence/absence of glasses.

```python
plt.figure(figsize=(20,5),dpi=50)
for i in range(4):
    ax = plt.subplot(1, 4, i + 1)
    p = i // 2         # Integer division to get the gender part (0 for female, 1 for male)
    q = i % 2          # Modulo operation to get the glasses part (0 for no glasses, 1 for glasses)

    z = z_female_g * p + z_male_g * (1 - p)  # Combine noise vectors based on gender
    label = labels_ng[0] * q + labels_g[0] * (1 - q)  # Combine labels based on presence/absence of glasses

    noise_and_labels = torch.cat(
        [z.reshape(1, z_dim, 1, 1), 
         label.reshape(1, 2, 1, 1)], dim=1).to(device)
    fake = generator(noise_and_labels)     # Generate an image using the combined noise and label

    img = (fake.cpu().detach()[0] / 2 + 0.5).permute(1, 2, 0)  # Normalize and permute for display
    plt.imshow(img.numpy())
    plt.xticks([])
    plt.yticks([])

plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
plt.show()
```

The logic here is that `p` determines the gender (female or male) from two possible noise vectors (`z_female_g` and `z_male_g`), while `q` decides whether to include glasses or not based on the labels. Combining these, you can generate images with both desired characteristics.

The resulting four cases will cover:
- Male faces with glasses
- Male faces without glasses
- Female faces with glasses
- Female faces without glasses

:p What are the values of `p` and `q`, and how do they determine the generated image's characteristics?
??x
The values of `p` and `q` are determined by integer division (`//`) and modulo (`%`) operations on `i`. Specifically:

- `p = i // 2`: This gives a value of either 0 or 1, representing whether to use `z_female_g` (if `p=0`) or `z_male_g` (if `p=1`).
- `q = i % 2`: This also gives a value of either 0 or 1, representing whether to include the glasses label (`labels_g[0]` if `q=1`) or not (`labels_ng[0]` if `q=0`).

By combining these values with the noise vectors and labels appropriately, you can generate images that reflect both gender and eyeglasses simultaneously.

For example:
- When `i = 0`, `p = 0 // 2 = 0` (female) and `q = 0 % 2 = 0` (no glasses). The image will be a female face without glasses.
- When `i = 1`, `p = 1 // 2 = 0` (female) and `q = 1 % 2 = 1` (glasses). The image will be a female face with glasses.
- When `i = 2`, `p = 2 // 2 = 1` (male) and `q = 2 % 2 = 0` (no glasses). The image will be a male face without glasses.
- When `i = 3`, `p = 3 // 2 = 1` (male) and `q = 3 % 2 = 1` (glasses). The image will be a male face with glasses.

This approach ensures that all four combinations of gender and presence/absence of eyeglasses are covered in the generated images.
x??

#### Noise Vector and Label Influence on Image Generation

Background context: In this scenario, we are working with a Conditional Generative Adversarial Network (cGAN) where we can control two independent characteristics of generated images—gender (male or female face) and the presence of eyeglasses. The noise vector and labels play key roles in generating different types of images.

:p How does the noise vector and label combination influence image generation in a cGAN?
??x
The noise vector and label combination significantly impact the type of image generated by the cGAN. Specifically, the generator takes two inputs: a noise vector \( z \) and a label indicating characteristics such as gender (male or female) and whether eyeglasses are present or not.

For example:
- If we feed a male face with glasses label to the model along with an appropriate noise vector, it will generate an image of a man wearing glasses.
- Similarly, a female face without glasses label combined with another suitable noise vector would produce an image of a woman without eyeglasses.

:p What is the significance of using different random noise vectors \( z_{female\_ng} \) and \( z_{male\_ng} \)?
??x
Using different random noise vectors for generating images based on gender (e.g., \( z_{female\_ng} \) and \( z_{male\_ng} \)) ensures that the model can generate diverse and realistic images for each characteristic independently. This allows us to see how slight variations in the input affect the output.

For instance, by using \( z_{female\_ng} \) or \( z_{male\_ng} \), we can observe different facial structures and features associated with females and males respectively.

:p How does label arithmetic contribute to generating images with varying characteristics?
??x
Label arithmetic enables us to interpolate between different labels, thereby generating a range of images that transition smoothly from one characteristic to another. This is particularly useful for exploring the space of generated images by varying the parameters in a controlled manner.

For example:
- Interpolating between \( z_{female\_ng} \) and \( z_{male\_ng} \) creates a sequence of images that change from male to female faces.
- Similarly, interpolating between labels with or without glasses can produce images where eyeglasses gradually appear or disappear.

:p How does vector arithmetic affect the generated images in a cGAN?
??x
Vector arithmetic allows us to interpolate between different noise vectors and labels, generating a series of intermediate images that transition from one state to another. This technique helps us understand how changes in input parameters influence the output images.

For instance, by interpolating \( z \) and label values:
- We can create a range of male faces with glasses gradually transitioning to those without.
- Similarly, for female faces, we can see a smooth transition from wearing glasses to not wearing them.

:p What is the code used to generate 36 images through vector arithmetic and label arithmetic?
??x
The provided code performs an interpolation between two noise vectors \( z_{female\_ng} \) and \( z_{male\_ng} \), as well as labels with or without glasses, generating a series of 36 images. Each image is generated by interpolating the values of \( p \) and \( q \).

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 20), dpi=50)
for i in range(36):
    ax = plt.subplot(6, 6, i + 1)
    
    p = i // 6
    q = i % 6
    
    z = z_female_ng * p / 5 + z_male_ng * (1 - p / 5)
    label = labels_ng[0] * q / 5 + labels_g[0] * (1 - q / 5)
    
    noise_and_labels = torch.cat(
        [z.reshape(1, z_dim, 1, 1),
         label.reshape(1, 2, 1, 1)], dim=1).to(device)
    
    fake = generator(noise_and_labels)
    img = (fake.cpu().detach()[0] / 2 + 0.5).permute(1, 2, 0)
    plt.imshow(img.numpy())
    plt.xticks([])
    plt.yticks([])

plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
plt.show()
```

- \( p \) and \( q \) are calculated based on the index `i`, allowing for a smooth transition.
- The noise vector and label values are interpolated linearly to create 6 different values each.
- This interpolation results in 36 unique combinations of images that vary smoothly from one characteristic (e.g., gender or glasses presence) to another.

x??

---

---

#### Interpolated Noise Vector and Label Arithmetic
Background context: In this project, you are working with a Conditional Generative Adversarial Network (cGAN) that generates images based on a combination of noise vectors and labels. The interpolated noise vector is a weighted average of two random noise vectors, which generate female and male faces respectively. Similarly, the label is a weighted average of labels indicating whether an image has eyeglasses or not.
:p How does the model use interpolation to create a series of images that transition from one characteristic to another?
??x
The model uses interpolation by taking a linear combination (weighted average) of two noise vectors and their corresponding labels. For example, given \( z_{\text{female}} \) and \( z_{\text{male}} \), the interpolated noise vector \( z_{\text{interpolated}} \) can be defined as:
\[ z_{\text{interpolated}} = \alpha \cdot z_{\text{female}} + (1 - \alpha) \cdot z_{\text{male}} \]
where \( 0 \leq \alpha \leq 1 \). Similarly, the label vector for eyeglasses can be interpolated as:
\[ l_{\text{interpolated}} = \alpha \cdot [0] + (1 - \alpha) \cdot [1] \]
or
\[ l_{\text{interpolated}} = \alpha \cdot [1] + (1 - \alpha) \cdot [0] \]
depending on the direction of transition. The trained model then generates 36 different images based on these interpolated vectors.

For each row, the eyeglasses label gradually changes from presence to absence as \( \alpha \) varies, and for each column, the face type (male to female or vice versa) transitions.
x??

---

#### Vector Arithmetic
Background context: The concept of vector arithmetic in this project refers to creating a series of images that transition between two different attributes by interpolating between noise vectors. Specifically, the model generates 36 images based on the interpolated noise vector and label, with each row showing a gradual change in eyeglasses presence or absence, while each column shows a transition from male to female faces.
:p How does vector arithmetic work in this context?
??x
Vector arithmetic works by taking linear combinations of two different noise vectors. Given \( z_{\text{female}} \) and \( z_{\text{male}} \), the interpolated noise vector can be defined as:
\[ z_{\text{interpolated}} = \alpha \cdot z_{\text{female}} + (1 - \alpha) \cdot z_{\text{male}} \]
where \( 0 \leq \alpha \leq 1 \). The model then generates images based on this interpolated noise vector with a corresponding label. For each row, the eyeglasses presence changes gradually from one extreme to another as \( \alpha \) varies, and for each column, the face type transitions between male and female.
x??

---

#### Label Arithmetic
Background context: Label arithmetic involves generating a series of images that transition from an image with one label (e.g., with glasses) to another (e.g., without glasses) by interpolating the labels. This is done by taking linear combinations of two different label vectors, such as:
\[ l_{\text{interpolated}} = \alpha \cdot [0] + (1 - \alpha) \cdot [1] \]
or
\[ l_{\text{interpolated}} = \alpha \cdot [1] + (1 - \alpha) \cdot [0] \]
where \( 0 \leq \alpha \leq 1 \).
:p How is label arithmetic used to generate a series of images that gradually transition from one state to another?
??x
Label arithmetic is applied by linearly interpolating between two labels. For instance, if we have two labels [0] and [1], representing the presence or absence of eyeglasses, an interpolated label can be created as:
\[ l_{\text{interpolated}} = \alpha \cdot [0] + (1 - \alpha) \cdot [1] \]
or
\[ l_{\text{interpolated}} = \alpha \cdot [1] + (1 - \alpha) \cdot [0] \]
where \( 0 \leq \alpha \leq 1 \). This means that as \( \alpha \) changes from 0 to 1, the label will transition smoothly between [0] and [1], representing a gradual change in eyeglasses presence or absence. The model then generates images with these interpolated labels, showing a series of transitions.
x??

---

#### Binary Label Representation
Background context: In this project, you are using binary values (1 and 0) to represent the presence or absence of eyeglasses instead of one-hot vectors. This simplifies the input data for the generator and critic models.
:p How can you modify the programs in this chapter to use binary labels instead of one-hot variables?
??x
To modify the programs to use binary values (1 and 0) instead of one-hot vectors, you need to adjust both the generator and critic input mechanisms. For example, if a label indicates the presence or absence of eyeglasses, you can use a single value:
- `1` for an image with glasses
- `0` for an image without glasses

Additionally, when feeding images to the critic, ensure that the fourth channel is filled appropriately based on whether the image has eyeglasses. If it does, fill the fourth channel with 0s; otherwise, fill it with 1s.

Here's a simplified example of how you might modify the input for the generator:
```java
// Pseudocode for modifying the generator input
public class Generator {
    // ...
    public void generateImage(double[] noiseVector) {
        double interpolatedLabel = 0.5; // Example value, can vary with alpha
        if (interpolatedLabel < 0.5) {
            // No glasses
            noiseVector[100] = 0; // Attach binary label to the noise vector
        } else {
            // Glasses
            noiseVector[100] = 1;
        }
        // Generate image using noiseVector and interpolatedLabel
    }
}
```

And for the critic:
```java
// Pseudocode for modifying the critic input
public class Critic {
    // ...
    public void evaluateImage(Image image) {
        double[] criticInput = new double[noiseLength + 1]; // +1 for the label
        // Fill in noise part of criticInput
        if (image.hasGlasses()) {
            criticInput[noiseLength] = 0; // Fourth channel is filled with 0s for glasses
        } else {
            criticInput[noiseLength] = 1; // Fourth channel is filled with 1s for no glasses
        }
        // Evaluate image using criticInput
    }
}
```

These modifications ensure that the model processes binary labels and correctly interprets the presence or absence of eyeglasses in input images.
x??

---

#### Wasserstein GAN (WGAN)
Background context: WGAN is a technique used to improve the training stability and performance of GAN models by using the Wasserstein distance as the loss function instead of the binary cross-entropy. The key idea is that the critic's function must be 1-Lipschitz continuous, meaning the gradient norms must be at most 1 everywhere.
:p What is the main difference between WGAN and traditional GANs?
??x
The main difference between WGAN and traditional GANs lies in their loss functions and training dynamics. In a standard GAN, the objective is to minimize the binary cross-entropy (cross-entropy) between real and generated data distributions. However, this often leads to issues like vanishing gradients or mode collapse.

In contrast, WGAN uses the Wasserstein distance as its loss function, which provides a more meaningful and stable measure of the difference between the generator's distribution and the true data distribution. The key requirement for WGAN is that the critic (discriminator) must be 1-Lipschitz continuous. This is achieved by adding a gradient penalty term to the loss function.

To enforce 1-Lipschitz continuity, the critic’s output should not change too rapidly with respect to its input. Specifically, the norm of the gradient of the critic's function should be bounded by 1. The gradient penalty term helps ensure this condition is met:
\[ \text{Loss} = -\mathbb{E}_{\hat{x} \sim D_{\text{real}}} [f(\hat{x})] + \mathbb{E}_{z \sim p(z)} [f(G(z))] + \lambda \cdot \sum_i (\|D(x_i)\|_2 - 1)^2 \]
where \( f \) is the critic, \( G \) is the generator, and \( \lambda \) is a hyperparameter that controls the penalty strength.

The gradient penalty ensures that small changes in input result in proportional changes in output, making the training more stable.
x??

#### CycleGAN Overview
CycleGAN was introduced in 2017 to enable image translation between two domains without paired data. This is particularly useful for tasks such as converting horse images to zebra images or transforming winter scenes into summer scenes.

:p What is the primary innovation of CycleGAN?
??x
The key innovation of CycleGAN lies in its ability to learn to translate between domains using unpaired examples, ensuring that both generators and discriminators are trained together. This method uses a cycle consistency loss function to ensure that an original image can be reconstructed from the transformed image.
```python
# Pseudocode for training CycleGAN
def train_cycle_gan(black_hair_images, blond_hair_images):
    # Initialize models
    black_hair_generator = BlackHairGenerator()
    blond_hair_generator = BlondHairGenerator()
    black_hair_discriminator = BlackHairDiscriminator()
    blond_hair_discriminator = BlondHairDiscriminator()

    # Define loss functions
    adversarial_loss = AdversarialLoss(black_hair_discriminator, blond_hair_discriminator)
    cycle_consistency_loss = CycleConsistencyLoss(blond_hair_generator, black_hair_generator)

    for epoch in range(num_epochs):
        for real_black_hair, real_blond_hair in zip(black_hair_images, blond_hair_images):
            # Train generators
            fake_blond_hair = black_hair_generator(real_black_hair)
            cycled_black_hair = black_hair_generator(fake_blond_hair)

            fake_black_hair = blond_hair_generator(real_blond_hair)
            cycled_blond_hair = blond_hair_generator(fake_black_hair)

            # Calculate losses
            total_loss_g = adversarial_loss(blond_hair_discriminator, real_blond_hair, fake_blond_hair) + cycle_consistency_loss(cycled_black_hair, real_black_hair)
            total_loss_g += adversarial_loss(black_hair_discriminator, real_black_hair, fake_black_hair) + cycle_consistency_loss(cycled_blond_hair, real_blond_hair)

            # Backpropagation and optimization
            optimizer_g.zero_grad()
            total_loss_g.backward()
            optimizer_g.step()

            # Train discriminators
            loss_d_blond = adversarial_loss(blond_hair_discriminator, real_blond_hair, fake_blond_hair)
            loss_d_black = adversarial_loss(black_hair_discriminator, real_black_hair, fake_black_hair)

            optimizer_d.zero_grad()
            (loss_d_blond + loss_d_black).backward()
            optimizer_d.step()

    return black_hair_generator, blond_hair_generator
```
x??

---

#### CycleGAN Generators and Discriminators
CycleGAN consists of two generators and two discriminators. Each generator is responsible for translating images from one domain to another, while the discriminators are used to determine if an image belongs to its respective domain.

:p What do the generators in a CycleGAN model do?
??x
The generators in a CycleGAN model convert images from one domain to another. For example, the black hair generator converts images with black hair into ones with blond hair, and the blond hair generator does the opposite.
```java
// Pseudocode for the Generators
public class Generator {
    // Generate fake image based on input real image
    public Image generateFakeImage(Image realImage) {
        // Apply transformations to create a fake image
        return transformedImage;
    }
}
```
x??

---

#### Cycle Consistency Loss
Cycle consistency loss ensures that the model preserves key features by ensuring the original image can be reconstructed from the transformed one after a round-trip conversion.

:p What is the purpose of cycle consistency loss in CycleGAN?
??x
The purpose of cycle consistency loss in CycleGAN is to ensure that the model can reconstruct the original image from its transformed version. This helps preserve important features during translation, ensuring that both generators and discriminators learn to maintain key characteristics.
```java
// Pseudocode for calculating cycle consistency loss
public class CycleConsistencyLoss {
    public double calculateCycleConsistencyLoss(Image realImage, Image cycledImage) {
        // Calculate difference between real image and cycled image
        return Math.abs(realImage - cycledImage);
    }
}
```
x??

---

#### Training Steps in CycleGAN
In each iteration of training, real images from both domains are fed into the generators to produce fake images. These fake images are then used as input for their respective discriminators, and losses are calculated.

:p Explain how CycleGAN minimizes cycle consistency losses.
??x
CycleGAN minimizes cycle consistency losses by ensuring that an original image can be reconstructed from the transformed one after a round-trip conversion. This is achieved by training both generators such that when a real black hair image goes through the blond hair generator and then back to the black hair generator, it should resemble the original as closely as possible.

This ensures that the model retains key features during translation:
1. Real black hair image → Black hair generator → Fake blond hair
2. Fake blond hair → Blond hair discriminator → Predict real or fake (Loss_D_Blond)
3. Fake blond hair → Black hair generator → Cycled back to real black hair (Cycled_black_hair)
4. Cycled_black_hair → Black hair discriminator → Predict original image (Loss_G_Black)

A similar process is repeated for the blond hair generator.
```java
// Pseudocode for training step
public void trainStep(Image realBlackHair, Image realBlondHair) {
    // Generate fake images
    Image fakeBlondHair = blackHairGenerator.generateRealToFake(realBlackHair);
    Image fakeBlackHair = blondHairGenerator.generateRealToFake(realBlondHair);

    // Cycled back to original
    Image cycledBlackHair = blackHairGenerator.generateFakeToReal(fakeBlondHair);
    Image cycledBlondHair = blondHairGenerator.generateFakeToReal(fakeBlackHair);

    // Calculate losses
    double lossD_Blond = adversarialLoss(blondHairDiscriminator, realBlondHair, fakeBlondHair);
    double cycleConsistencyLoss_Black = cycleConsistencyLoss(cycledBlackHair, realBlackHair);
    double lossD_Black = adversarialLoss(blackHairDiscriminator, realBlackHair, fakeBlackHair);
    double cycleConsistencyLoss_Blond = cycleConsistencyLoss(cycledBlondHair, realBlondHair);

    // Update model parameters
    blackHairGenerator.updateParams(lossD_Black + cycleConsistencyLoss_Black);
    blondHairGenerator.updateParams(lossD_Blond + cycleConsistencyLoss_Blond);
}
```
x??

---

