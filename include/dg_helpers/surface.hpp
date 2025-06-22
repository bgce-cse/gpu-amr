// Placeholder for your 2D array type, replace with Eigen or your own matrix type
template<typename T>
using Matrix2D = /* your matrix type */;

template<typename Basis, typename Eq, typename Globals, typename Cell, typename Scalar = double>
class BuffersFaceIntegral {
private:
    Matrix2D<Scalar> dofsface_;
    Matrix2D<Scalar> dofsfaceneigh_;
    Matrix2D<Scalar> fluxface_;
    Matrix2D<Scalar> fluxfaceneigh_;
    Matrix2D<Scalar> numericalflux_;
    Matrix2D<Scalar> numericalflux_scaled_;

public:
    BuffersFaceIntegral(const Basis& basis, size_t ndofs) {
        assert(basis.dimensions() == 2);

        size_t basissize_1d = basis.size_1d();
        // basis.size() if needed

        dofsface_ = Matrix2D<Scalar>(basissize_1d, ndofs);
        dofsfaceneigh_ = Matrix2D<Scalar>(basissize_1d, ndofs);
        fluxface_ = Matrix2D<Scalar>(basissize_1d * 2, ndofs);
        fluxfaceneigh_ = Matrix2D<Scalar>(basissize_1d * 2, ndofs);
        numericalflux_ = Matrix2D<Scalar>(basissize_1d, ndofs);
        numericalflux_scaled_ = Matrix2D<Scalar>(basissize_1d, ndofs);

        // Initialize with zeros if needed
    }

    // Rusanov numerical flux calculation
    Scalar rusanov(
        const Eq& eq,
        const Matrix2D<Scalar>& dofs,
        const Matrix2D<Scalar>& dofsneigh,
        const Matrix2D<Scalar>& flux,
        const Matrix2D<Scalar>& fluxneigh,
        Scalar dx,
        int normalidx,
        Scalar normalsign,
        Matrix2D<Scalar>& numericalflux
    ) {
        Scalar maxeigenval_center = max_eigenval(eq, dofs, normalidx);
        Scalar maxeigenval_neigh = max_eigenval(eq, dofsneigh, normalidx);
        Scalar maxeigenval = std::max(maxeigenval_center, maxeigenval_neigh);

        size_t basissize_1d = dofs.rows();

        Matrix2D<Scalar> local_flux_normal_component;
        Matrix2D<Scalar> local_fluxneigh_normal_component;

        if (normalidx == 1) {
            // Select rows [0, basissize_1d) for flux and fluxneigh
            local_flux_normal_component = flux.block(0, 0, basissize_1d, flux.cols());
            local_fluxneigh_normal_component = fluxneigh.block(0, 0, basissize_1d, fluxneigh.cols());
        }
        else if (normalidx == 2) {
            // Select rows [basissize_1d, end) for flux and fluxneigh
            local_flux_normal_component = flux.block(basissize_1d, 0, basissize_1d, flux.cols());
            local_fluxneigh_normal_component = fluxneigh.block(basissize_1d, 0, basissize_1d, fluxneigh.cols());
        }
        else {
            throw std::runtime_error("Invalid normalidx: Expected 1 or 2 for 2D simulation.");
        }

        Matrix2D<Scalar> first_term = 0.5 * normalsign * (local_flux_normal_component + local_fluxneigh_normal_component);
        Matrix2D<Scalar> second_term = 0.5 * maxeigenval * (dofs - dofsneigh);

        // If eq is Acoustic (you need to implement a check or a trait)
        if constexpr (std::is_same_v<Eq, Acoustic>) {
            // zero columns 3 and 4 (zero-based indices 3,4) in second_term
            // This is a placeholder, adapt to your matrix type
            for (size_t col = 3; col <= 4; ++col) {
                for (size_t row = 0; row < second_term.rows(); ++row) {
                    second_term(row, col) = 0;
                }
            }
        }

        numericalflux = dx * (first_term + second_term);

        return maxeigenval;
    }

    // Project dofs and flux to a face
    void project_to_faces(
        const Globals& globals,
        const Matrix2D<Scalar>& dofs,
        const Matrix2D<Scalar>& flux,
        Matrix2D<Scalar>& dofsface,
        Matrix2D<Scalar>& fluxface,
        int face
    ) {
        dofsface = globals.project_dofs_to_face(face) * dofs;
        fluxface = globals.project_flux_to_face(face) * flux;
    }

    // Evaluate face integral and update celldu
    Scalar evaluate_face_integral(
        const Eq& eq,
        const Globals& globals,
        const Cell& cell,
        int face,
        Matrix2D<Scalar>& celldu
    ) {
        int normalidx = globals.normalidxs(face);
        Scalar normalsign = globals.normalsigns(face);

        // Reset numericalflux buffer
        numericalflux_.setZero();

        Scalar maxeigenval = rusanov(
            eq, dofsface_, dofsfaceneigh_, fluxface_, fluxfaceneigh_,
            cell.size(0), normalidx, normalsign, numericalflux_
        );

        celldu -= globals.project_dofs_from_face(face) * (numericalflux_.transpose() * globals.reference_massmatrix_face()).transpose();

        return maxeigenval;
    }

    // Getters for buffers if needed externally
    Matrix2D<Scalar>& dofsface() { return dofsface_; }
    Matrix2D<Scalar>& dofsfaceneigh() { return dofsfaceneigh_; }
    Matrix2D<Scalar>& fluxface() { return fluxface_; }
    Matrix2D<Scalar>& fluxfaceneigh() { return fluxfaceneigh_; }
    Matrix2D<Scalar>& numericalflux() { return numericalflux_; }
    Matrix2D<Scalar>& numericalflux_scaled() { return numericalflux_scaled_; }
};