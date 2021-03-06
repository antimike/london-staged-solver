//For grid visualization
#include <deal.II/grid/grid_out.h>

//Includes copied from vector-poisson-solver
#include <deal.II/base/tensor.h>
#include <boost/math/constants/constants.hpp>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <iostream>
#include <fstream>

namespace LondonStagedSolver
{
  using namespace dealii;

  template<int dim>
    class LondonStagedSolverClass
    {
      public:
        LondonStagedSolverClass(const unsigned int completeness_degree);
        void run();
        void print_grids();

      private:

        // This strategy of keeping track of domain IDs is taken from step-46
        enum 
        {
          sample_id,
          exterior_vacuum_id,
          core_upper_id,
          core_lower_id,
        };

        // Degree of finite elements to use
        const unsigned int completeness_degree;

        // Convenience wrapper around material id enum
        static bool cell_is_in_sample(
            const typename hp::DoFHandler<dim>::cell_iterator &cell);

        // Create the initial mesh
        void make_grid();

        // Sets the "active indices" (i.e., identifies which function space is applicable) on each cell
        void setup_active_fe_indices();

        // Initializes containers, distributes DOFs, and calculates constraints.
        void setup_system();

        // Calculates the system matrix and RHS from the weak form of the problem.
        void assemble_system();

        // Solves the linear system.
        void solve();

        // Estimates errors and marks cells for refinement.
        void refine_grid();

        // Writes results to files for visualization.
        void output_results(const unsigned int cycle) const;

        // Container for the mesh (discretization).
        Triangulation<dim> triangulation;

        //We need to use collections from the 'hp' namespace because the elements may be region-specific.
        hp::FECollection<dim> fe_collection;
        hp::DoFHandler<dim> dof_handler;

        // The finite-element systems for the internal and external function spaces
        FESystem<dim>   int_fe;
        FESystem<dim>   ext_fe;

        // This is the container which will end up holding all the constraints in the problem
        // (i.e., boundary conditions and "unphysical" results of uneven mesh refinement called hanging nodes.)
        AffineConstraints<double> constraints;

        // A container for the sparse linear system which will eventually result from the weak form of our problem.
        SparseMatrix<double> system_matrix;

        // Necessary to set up the SparseMatrix.  Basically this just records where the nonzero entries are.
        SparsityPattern      sparsity_pattern;

        // A container for the solution (the vector potential in this case).
        Vector<double> solution;

        // A container for the RHS of the linear system.
        Vector<double> system_rhs;

        // Tunable constant used to enforce gauge condition on A-star
        const double penalty_constant;
    };

  template <int dim>
    LondonStagedSolverClass<dim>::LondonStagedSolverClass(const unsigned int completeness_degree)
    : completeness_degree(penalty_constant),
    penalty_constant(1.0),
    triangulation(Triangulation<dim>::maximum_smoothing),
    dof_handler(triangulation),
    int_fe(FE_Nothing<dim>(),     // Psi (magnetic scalar potential) is not needed inside the sample
        1,
        FE_Q<dim>(completeness_degree),
        dim,                      // A* (vector potential)---needed in both regions
        FE_Q<dim>(completeness_degree),
        1                         // theta---only needed inside the sample
        ),
    ext_fe(FE_Q<dim>(completeness_degree),
        1,                   // Psi
        FE_Q<dim>(completeness_degree),
        dim,                    // A*
        FE_Nothing<dim>(),         // theta
        1
        )
  {
    // Push the finite elements into the HP collection
    // Has to be done in the same order as the material IDs appear in the enum
    fe_collection.push_back(int_fe);
    fe_collection.push_back(ext_fe);
  }

  template <int dim>
    bool LondonStagedSolverClass<dim>::cell_is_in_sample(
        const typename hp::DoFHandler<dim>::cell_iterator &cell)
    {
      return (cell->material_id() == sample_id);
    }

  // Makes initial grid
  // Currently consists of a cylindrical shell inside a larger hypercube (to represent the surrounding vacuum)
  template <int dim>
    void LondonStagedSolverClass<dim>::make_grid()
    {
      const double cube_length = 8;
      const int initial_cube_granularity = 8;
      const double cylinder_length = 6;
      const double inner_radius = 1;
      const double outer_radius = 3;

      Triangulation<dim> sample;
      Triangulation<dim> vacuum;
      Triangulation<dim> core_upper;
      Triangulation<dim> core_lower;
      Triangulation<dim> sample_plus_core;

      // Creates a hypercube of length cube_length centered on origin
      // "initial_cube_granularity" sets the number of initial subdivisions
      GridGenerator::subdivided_hyper_cube(
          vacuum, 
          initial_cube_granularity, 
          -cube_length / 2, 
          cube_length / 2);

      // Creates cylindrical shell centered on z-axis with end-faces at z = 0, z = L
      GridGenerator::cylinder_shell(
          sample,
          cylinder_length,
          inner_radius,
          outer_radius
          );

      // Shift the cylinder to be centered on the origin with axis parallel to the x-axis
      GridTools::shift(
          //new Tensor<1, dim>(0, 0, -cylinder_length/2),
          Point<dim>(0, 0, -cylinder_length/2),
            sample
          );
      GridTools::rotate(
          boost::math::double_constants::pi/2,
          2,       // Rotate about y-axis
          sample);

      // Upper part of "core" region (cylindrical hole)
      // Centered on x-axis
      GridGenerator::cylinder(
          core_upper,
          inner_radius,
          cylinder_length/4
          );

      // Lower part of "core" region
      // Centered on x-axis
      GridGenerator::cylinder(
          core_lower,
          inner_radius,
          cylinder_length/4
          );

      // Shift the core triangulations to the x > 0 and x < 0 core regions (respectively)
      GridTools::shift(
          //new Tensor<1, dim>(cylinder_length/4, 0, 0),
          Point<3>(cylinder_length/4, 0, 0),
          core_upper
          );
      GridTools::shift(
          //new Tensor<1, dim>(-cylinder_length/4, 0, 0),
          Point<3>(-cylinder_length/4, 0, 0),
          core_lower
          );
      //GridGenerator::create_triangulation_with_removed_cells(sample, core_upper, spacedim>::active_cell_iterator> &cells_to_remove, Triangulation<dim, spacedim> &result)

      // Set material IDs
      // Different material IDs in the vacuum region will be used to identify faces on the 
      // "cut surface" which generates the boundary / source terms for flux quantization
      for(auto &cell: sample.active_cell_iterators())
      {
        cell->set_material_id(sample_id);
      }
      for(auto &cell: core_upper.active_cell_iterators())
      {
        cell->set_material_id(core_upper_id);
      }
      for(auto &cell: core_lower.active_cell_iterators())
      {
        cell->set_material_id(core_lower_id);
      }

      // Merge triangulations
      GridGenerator::merge_triangulations(sample, core_lower, sample_plus_core);
      GridGenerator::merge_triangulations(sample_plus_core, core_upper, sample_plus_core);
      //GridGenerator::create_triangulation_with_removed_cells(vacuum, sample_plus_core, vacuum);
      // Not sure if this has to be done after the above subtraction operation, but it can't hurt
      for(auto &cell: vacuum.active_cell_iterators())
      {
        cell->set_material_id(exterior_vacuum_id);
      }
      GridGenerator::merge_triangulations(sample_plus_core, vacuum, triangulation);
    }

  template<int dim>
    void LondonStagedSolverClass<dim>::print_grids()
    {
      make_grid();
      std::ofstream out("triangulation.svg");
      GridOut grid_out;
      grid_out.write_svg(triangulation, out);

      std::cout << "Triangulation written to 'triangulation.svg'" << std::endl;
    }

  // Taken from step-46
  // This relies on the material IDs set in the function make_grid() to apply the correct FE function space to each cell
  template <int dim>
  void LondonStagedSolverClass<dim>::setup_active_fe_indices()
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell_is_in_sample(cell))
          cell->set_active_fe_index(sample_id);
        else 
          cell->set_active_fe_index(exterior_vacuum_id);
        //else
          //Assert(false, ExcNotImplemented());
      }
  }

  template <int dim>
  void LondonStagedSolverClass<dim>::setup_system()
  {
    setup_active_fe_indices();
    dof_handler.distribute_dofs(fe_collection);

    {
      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      const FEValuesExtractors::Scalar magnetic_scalar_potential(0);
      //VectorTools::interpolate_boundary_values(dof_handler,
                                               //1,
                                               //StokesBoundaryValues<dim>(),
                                               //constraints,
                                               //fe_collection.component_mask(
                                                 //magnetic_scalar_potential));

      const FEValuesExtractors::Vector magnetic_vector_potential(1);
      //VectorTools::interpolate_boundary_values(
        //dof_handler,
        //0,
        //Functions::ZeroFunction<dim>(dim + 1 + dim),
        //constraints,
        //fe_collection.component_mask(magnetic_vector_potential));
      const FEValuesExtractors::Scalar wavefunction_phase(dim + 1);
    }

    // There are more constraints we have to handle, though: we have to make
    // sure that the velocity is zero at the interface between fluid and
    // solid. The following piece of code was already presented in the
    // introduction:
/*
 *    {
 *      std::vector<types::global_dof_index> local_face_dof_indices(
 *        stokes_fe.dofs_per_face);
 *      for (const auto &cell : dof_handler.active_cell_iterators())
 *        if (cell_is_in_fluid_domain(cell))
 *          for (unsigned int face_no : GeometryInfo<dim>::face_indices())
 *            {
 *              bool face_is_on_interface = false;
 *
 *              if ((cell->neighbor(face_no)->has_children() == false) &&
 *                  (cell_is_in_solid_domain(cell->neighbor(face_no))))
 *                face_is_on_interface = true;
 *              else if (cell->neighbor(face_no)->has_children() == true)
 *                {
 *                  for (unsigned int sf = 0;
 *                       sf < cell->face(face_no)->n_children();
 *                       ++sf)
 *                    if (cell_is_in_solid_domain(
 *                          cell->neighbor_child_on_subface(face_no, sf)))
 *                      {
 *                        face_is_on_interface = true;
 *                        break;
 *                      }
 *                }
 *
 *              if (face_is_on_interface)
 *                {
 *                  cell->face(face_no)->get_dof_indices(local_face_dof_indices,
 *                                                       0);
 *                  for (unsigned int i = 0; i < local_face_dof_indices.size();
 *                       ++i)
 *                    if (stokes_fe.face_system_to_component_index(i).first < dim)
 *                      constraints.add_line(local_face_dof_indices[i]);
 *                }
 *            }
 *    }
 */

    // At the end of all this, we can declare to the constraints object that
    // we now have all constraints ready to go and that the object can rebuild
    // its internal data structures for better efficiency:
    constraints.close();

    std::cout << "   Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    // In the rest of this function we create a sparsity pattern as discussed
    // extensively in the introduction, and use it to initialize the matrix;
    // then also set vectors to their correct sizes:
    {
      DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

      Table<2, DoFTools::Coupling> cell_coupling(fe_collection.n_components(),
                                                 fe_collection.n_components());
      Table<2, DoFTools::Coupling> face_coupling(fe_collection.n_components(),
                                                 fe_collection.n_components());

      for (unsigned int c = 0; c < fe_collection.n_components(); ++c)
        for (unsigned int d = 0; d < fe_collection.n_components(); ++d)
          {
            if (((c < dim + 1) && (d < dim + 1) &&
                 !((c == dim) && (d == dim))) ||
                ((c >= dim + 1) && (d >= dim + 1)))
              cell_coupling[c][d] = DoFTools::always;

            if ((c >= dim + 1) && (d < dim + 1))
              face_coupling[c][d] = DoFTools::always;
          }

      DoFTools::make_flux_sparsity_pattern(dof_handler,
                                           dsp,
                                           cell_coupling,
                                           face_coupling);
      constraints.condense(dsp);
      sparsity_pattern.copy_from(dsp);
    }

    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }
}

int main() {
  try 
  {
    using namespace LondonStagedSolver;

    LondonStagedSolverClass<3> staged_solver(2);
    staged_solver.print_grids();
  }
  catch(std::exception &exc) 
  {
    std::cerr << std::endl
      << std::endl
      << "----------------------------------------------------"
      << std::endl;
    std::cerr << "Exception on processing: " << std::endl
      << exc.what() << std::endl
      << "Aborting!" << std::endl
      << "----------------------------------------------------"
      << std::endl;

    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
      << std::endl
      << "----------------------------------------------------"
      << std::endl;
    std::cerr << "Unknown exception!" << std::endl
      << "Aborting!" << std::endl
      << "----------------------------------------------------"
      << std::endl;
    return 1;
  }
  return 0;
}
