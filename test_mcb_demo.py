"""Demo: Using the MCB forcing module."""                                                                                                                                         
                                                                                                                                                                                    
import jax.numpy as jnp                                                                                                                                                           
from jcm.mcb import (                                                                                                                                                             
      MCBConfig,                                                                                                                                                                    
      create_stratocumulus_mask,                                                                                                                                                    
      compute_mcb_sea_albedo,                                                                                                                                                       
      STRATOCUMULUS_REGIONS,                                                                                                                                                        
  )                                                                                                                                                                                 
from jcm.physics.speedy.speedy_coords import get_speedy_coords                                                                                                                    
                                                                                                                                                                                    
# Setup coordinates                                                                                                                                                               
coords = get_speedy_coords(layers=8, spectral_truncation=21)                                                                                                                      
grid = coords.horizontal                                                                                                                                                          
nodal_shape = grid.nodal_shape                                                                                                                                                    
                                                                                                                                                                                
print(f"Grid shape: {nodal_shape}")  # (64, 32)                                                                                                                                   
                                                                                                                                                                                
# Create a fake land-sea mask (all ocean for demo)                                                                                                                                
fmask = jnp.zeros(nodal_shape)                                                                                                                                                    
                                                                                                                                                                                
# Create MCB deployment mask for SE Pacific                                                                                                                                       
se_pacific_mask = create_stratocumulus_mask(grid, fmask, ['se_pacific'])                                                                                                          
print(f"SE Pacific mask covers {jnp.sum(se_pacific_mask):.0f} grid cells")                                                                                                        
                                                                                                                                                                                
# Create MCB configuration: 5% albedo increase in SE Pacific                                                                                                                      
mcb_config = MCBConfig.uniform(                                                                                                                                                   
    nodal_shape,                                                                                                                                                                  
    perturbation=0.05,                                                                                                                                                            
    region_mask=se_pacific_mask,                                                                                                                                                  
)                                                                                                                                                                                 
                                                                                                                                                                                
# Compute effective albedo with MCB                                                                                                                                               
base_albedo = 0.07  # Normal ocean albedo                                                                                                                                         
mcb_albedo = compute_mcb_sea_albedo(mcb_config, base_albedo, day_of_year=180)                                                                                                     
                                                                                                                                                                                
print(f"Base ocean albedo: {base_albedo}")                                                                                                                                        
print(f"MCB albedo range: [{float(jnp.min(mcb_albedo)):.3f}, {float(jnp.max(mcb_albedo)):.3f}]")                                                                                  
print(f"Max albedo increase: {float(jnp.max(mcb_albedo) - base_albedo):.3f}")