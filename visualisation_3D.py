import open3d as o3d
from utils import load_xyz
from pathlib import Path


# === Dictionnaire simple pour les couleurs par type d'atome ===
atom_colors = {
    'H': [1, 1, 1],        # blanc
    'C': [0.2, 0.2, 0.2],  # gris foncé
    'O': [1, 0, 0],        # rouge
    'N': [0, 0, 1],        # bleu
    'S': [1, 1, 0],        # jaune
    # ajoute d'autres si besoin
}

class Visualisation3D:

    def __init__(self, xyz_dir):
        self.xyz_dir = xyz_dir

    # === Fonction pour créer une sphère 3D pour un atome ===
    def create_atom_sphere(self, position, radius=0.3, color=[0.5, 0.5, 0.5]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(position.tolist())
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        return sphere

    # === Fonction d'affichage d'une molécule ===
    def visualize_molecule(self, atom_types, positions):
        geometries = []
        for i, (atype, pos) in enumerate(zip(atom_types, positions)):
            color = atom_colors.get(atype, [0.5, 0.5, 0.5])  # couleur grise par défaut
            sphere = self.create_atom_sphere(pos, radius=0.3, color=color)
            geometries.append(sphere)
            

        o3d.visualization.draw_geometries(geometries)


    def visualize_depuis_id(self, mol_id):
        xyz_file = self.xyz_dir + f"id_{mol_id}.xyz"
        atom_types, positions = load_xyz(xyz_file)
        print(f"Visualisation de la molécule {mol_id} avec {len(atom_types)} atomes")
        self.visualize_molecule(atom_types, positions)