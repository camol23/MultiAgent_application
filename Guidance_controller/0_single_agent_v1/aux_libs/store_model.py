import torch
import torch.nn as nn
import os




def save_model(model, optimizer, episode, reward_history, path="checkpoints", file_name="checkpoint_episode_"):
    """
        Save the DRL model, optimizer state, and training history
        
        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            episode: Current episode number
            reward_history: List of rewards
            path: Directory to save the checkpoint
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'reward_history': reward_history
    }
    
    # checkpoint_path = os.path.join(path, f'checkpoint_episode_{episode}.pt')
    checkpoint_path = os.path.join(path, file_name + str(episode) + ".pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")



def load_model(model, optimizer, checkpoint_path):
    """
        Load a saved DRL model and its training state

        Args:
            model: PyTorch model instance to load weights into
            optimizer: Optimizer instance to load state into
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            model: Loaded model
            optimizer: Loaded optimizer
            episode: Episode number when checkpoint was saved
            reward_history: History of rewards
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return (
        model,
        optimizer,
        checkpoint['episode'],
        checkpoint['reward_history']
    )