"""Demo: Verify gradients flow through MCB."""                                                                                                                   
                                                                                                                                                                   
import jax                                                                                                                                                       
import jax.numpy as jnp                                                                                                                                          
from jcm.mcb import MCBConfig, compute_mcb_sea_albedo                                                                                                            
                                                                                                                                                                
nodal_shape = (64, 32)                                                                                                                                           
                                                                                                                                                                
# Create MCB config                                                                                                                                              
config = MCBConfig.uniform(nodal_shape, perturbation=0.05)                                                                                                       
                                                                                                                                                                
# Define a simple loss function                                                                                                                                  
def loss_fn(albedo_perturbation):                                                                                                                                
    cfg = config.copy(albedo_perturbation=albedo_perturbation)                                                                                                   
    effective_albedo = compute_mcb_sea_albedo(cfg, 0.07, day_of_year=100)                                                                                        
    # Loss = mean albedo (we could minimize or maximize this)                                                                                                    
    return jnp.mean(effective_albedo)                                                                                                                            
                                                                                                                                                                
# Compute gradient                                                                                                                                               
grad = jax.grad(loss_fn)(config.albedo_perturbation)                                                                                                             
                                                                                                                                                                
print(f"Gradient shape: {grad.shape}")                                                                                                                           
print(f"Gradient has NaNs: {bool(jnp.any(jnp.isnan(grad)))}")                                                                                                    
print(f"Gradient is nonzero: {bool(jnp.any(grad != 0))}")                                                                                                        
print(f"Gradient mean: {float(jnp.mean(grad)):.6f}")       