"""Demo: Run JCM with MCB physics enabled."""                                                                                                                    
                                                                                                                                                                   
from jcm.model import Model                                                                                                                                      
from jcm.physics.speedy.speedy_physics import SpeedyPhysics                                                                                                      
from jcm.physics.speedy.speedy_coords import get_speedy_coords                                                                                                   
from jcm.terrain import TerrainData                                                                                                                              
from jcm.forcing import default_forcing                                                                                                                          
from jcm.mcb import MCBConfig                                                                                                                                    
import jax.numpy as jnp                                                                                                                                          
                                                                                                                                                                  
# Setup                                                                                                                                                          
coords = get_speedy_coords(layers=8, spectral_truncation=21)                                                                                                     
terrain = TerrainData.aquaplanet(coords)                                                                                                                         
forcing = default_forcing(coords.horizontal)                                                                                                                     
                                                                                                                                                                  
# Create MCB config                                                                                                                                              
mcb_config = MCBConfig.uniform(                                                                                                                                  
    coords.horizontal.nodal_shape,                                                                                                                               
    perturbation=0.05,                                                                                                                                           
)                                                                                                                                                                
                                                                                                                                                                  
# Create model WITH MCB                                                                                                                                          
physics_with_mcb = SpeedyPhysics(mcb_config=mcb_config)                                                                                                          
model = Model(coords=coords, terrain=terrain, physics=physics_with_mcb)                                                                                          
                                                                                                                                                                  
# Run short simulation (1 day)                                                                                                                                   
predictions = model.run(forcing=forcing, save_interval=1.0, total_time=1.0)                                                                                      
                                                                                                                                                                  
print("Model ran successfully with MCB!")                                                                                                                        
print(f"Output temperature shape: {predictions.dynamics.temperature.shape}") 