"""Warp contact constraints against the unchanged MHR triangle mesh.

Supplement VBD with Jacobi constraints at cloth vertices, edge quarters and
face quadrature points after every substep. This is a contact portion of the
time step, not export-time shrinkwrap. Unsupported cloth remains free.
"""
import numpy as np
import warp as wp


@wp.kernel
def contact_samples(body: wp.uint64, positions: wp.array[wp.vec3], inv_mass: wp.array[float],
                    indices: wp.array2d[int], weights: wp.array2d[float], margin: float,
                    deltas: wp.array[wp.vec3], counts: wp.array[int]):
    sample = wp.tid()
    p = wp.vec3(0.0)
    denom = float(0.0)
    for k in range(3):
        vertex = indices[sample, k]
        weight = weights[sample, k]
        p += positions[vertex] * weight
        denom += inv_mass[vertex] * weight * weight
    if denom > 0.0:
        query = wp.mesh_query_point_sign_normal(body, p, 0.12)
        if query.result:
            closest = wp.mesh_eval_position(body, query.face, query.u, query.v)
            distance = wp.length(p - closest)
            signed = distance * query.sign
            if signed < margin:
                normal = wp.mesh_eval_face_normal(body, query.face)
                if distance > 1.0e-7:
                    normal = (p - closest) * (query.sign / distance)
                penetration = wp.min(margin - signed, 0.02)
                for k in range(3):
                    vertex = indices[sample, k]
                    weight = weights[sample, k]
                    if weight > 0.0 and inv_mass[vertex] > 0.0:
                        correction = normal * (penetration * inv_mass[vertex] * weight / denom)
                        wp.atomic_add(deltas, vertex, correction)
                        wp.atomic_add(counts, vertex, 1)


@wp.kernel
def apply_contacts(positions: wp.array[wp.vec3], velocities: wp.array[wp.vec3],
                   deltas: wp.array[wp.vec3], counts: wp.array[int]):
    i = wp.tid()
    if counts[i] > 0:
        correction = deltas[i] / float(counts[i])
        positions[i] += correction
        length = wp.length(correction)
        if length > 1.0e-8:
            normal = correction / length
            velocity = velocities[i]
            vn = wp.dot(velocity, normal)
            if vn < 0.0:
                velocity -= vn * normal
            velocities[i] = velocity * 0.98


@wp.kernel
def damp_velocity(velocities: wp.array[wp.vec3]):
    i = wp.tid()
    velocities[i] *= 0.97


class BodyContact:
    def __init__(self, body, garments, model):
        self.body_mesh = wp.Mesh(points=wp.array(body.vertices.astype(np.float32), dtype=wp.vec3, device=model.device),
                                 indices=wp.array(body.faces.flatten().astype(np.int32), dtype=int, device=model.device))
        indices, weights = [], []
        offset = 0
        for garment in garments:
            mesh = garment['mesh']
            for i in range(len(mesh.vertices)):
                indices.append([offset+i]*3); weights.append([1,0,0])
            for a,b in mesh.edges_unique:
                for t in (.25,.5,.75):
                    indices.append([offset+int(a),offset+int(b),offset+int(b)])
                    weights.append([1-t,t,0])
            for face in mesh.faces:
                for bary in ((1/3,1/3,1/3),(.6,.2,.2),(.2,.6,.2),(.2,.2,.6)):
                    indices.append((face+offset).tolist()); weights.append(bary)
            offset += len(mesh.vertices)
        self.indices = wp.array(np.asarray(indices,dtype=np.int32),dtype=int,device=model.device)
        self.weights = wp.array(np.asarray(weights,dtype=np.float32),dtype=float,device=model.device)
        self.inv_mass = model.particle_inv_mass
        self.deltas = wp.zeros(model.particle_count,dtype=wp.vec3,device=model.device)
        self.counts = wp.zeros(model.particle_count,dtype=int,device=model.device)
        self.count = len(indices)
        self.vertex_count = model.particle_count

    def solve(self, state, iterations=3):
        for _ in range(iterations):
            self.deltas.zero_(); self.counts.zero_()
            wp.launch(contact_samples,dim=self.count,inputs=[self.body_mesh.id,state.particle_q,self.inv_mass,
                self.indices,self.weights,.002,self.deltas,self.counts],device=state.particle_q.device)
            wp.launch(apply_contacts,dim=self.vertex_count,inputs=[state.particle_q,state.particle_qd,
                self.deltas,self.counts],device=state.particle_q.device)
        wp.launch(damp_velocity,dim=self.vertex_count,inputs=[state.particle_qd],device=state.particle_q.device)
