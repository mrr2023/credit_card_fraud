import matplotlib.pyplot as plt
import numpy as np
import torch

    
def train(training_data, validation_data, model, model_name,
          n_epochs=10, lr=.001, gamma=.25, step_size=None, 
          clip_val=None, loss_weights=None):

    # Idea here is to correct for imbalanced data in the loss function
    if loss_weights is not None:
        loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(loss_weights))
    else:
        loss = torch.nn.CrossEntropyLoss()
        
    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # Case when we want to use a learning rate scheduler
    if step_size is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Will be used for logging purposes
    loss_arr = []

    # Run SGD for several epochs
    for epoch in range(n_epochs):

        # Specifiy that we are in training mode
        model.train()

        # Iterate through the batches
        for X, y in training_data:

            # Zero out old gradient calculations before backpropagating based on new batch
            optimizer.zero_grad()

            # Compute the loss on the given batch
            preds = model(X)
            batch_loss = loss(preds, y)

            # Backpropagate loss
            batch_loss.backward()

            # Add gradient clipping
            if clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

            # Update parameters
            optimizer.step()

        # Assess updates on validation data
        model.eval()
        epoch_loss_list = []

        for X, y in validation_data:
            preds = model(X).detach()
            epoch_loss_list.append(loss(preds, y).float())

        # See performance of the model at the end of the epoch (every 50 epochs)
        if epoch % 100 == 0:
            print(f'loss at epoch {epoch}: {np.mean(epoch_loss_list)}')
            
        loss_arr.append(np.mean(epoch_loss_list))

        # Save a dictionary of the model's parameters at the end of each epoch
        torch.save(model.state_dict(), f'model_files/{model_name}.th')

    # Visualize outputs
    plt.figure(figsize=(14, 6))  
    plt.title('Validation Loss as a Function of Epoch', fontsize=14)
    plt.plot([_ for _ in range(len(loss_arr))], loss_arr)