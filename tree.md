Implement get_value<QOI>(tree, neigbour, [](){} interpolation startegy) or
similar

Iterate over the linear index map in the solver or whatever

Store topology and type in the tree

Boundary values
  - Computed at runtime through lambdas
  - Update the boundaries inside the domain (staggered grid)
  - Internal boundaries inside the domain
  - Apply boundary conditions to this ones too
  
Initial refinement:
  - Define kernel that returns for each cell one of {refine, fluid, obstacle} or
    something like this.

Idea:
  Define an API that would work for the tree.
  Implement get_value trivially since 
  Make matrices global for now, so that they can be accessed through free functions.

API:
  enum struct Direction{
    None, N, E, S, W
  }
  
  typedef pressure_t = strcuct pressure{ typnename type = double; };
  typedef velocity_u_t = strcuct velocity_u{ typnename type = double; };
  typedef velocity_v_t = strcuct velocity_v{ typnename type = double; };
  typedef flux_f_t = strcuct flux_f{ typnename type = double; };
  typedef flux_g_t = strcuct flux_g{ typnename type = double; };
  typedef temperature_t = strcuct temetrature_t{ typnename type = double; };

  // Free function
  // Returns value at the requested side, boudnaries too, interpolated already.
  get_value<Field>(int i, int j, Direction d) -> typename Field::type  (return double basically)

Implement this using matrices for now basically
use if constexpr if to choose matrix:

update indices:
if d == Direction::E:  i += 1; break;
if d == Direction::N:  j += 1; break;
if d == Direction::W:  i -= 1; break;
if d == Direction::S:  j -= 1; break;
if d == Direction::None: break; // Return current posisiton basically.

if constexpr (std::is_same_v<Field, pressure_t>){
  return pressure_matrix(i , j);
}
repeat ...


