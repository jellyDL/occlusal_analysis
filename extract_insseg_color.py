import os
import numpy as np
import open3d as o3d
from collections import Counter
import trimesh
import tempfile

def analyze_ply_colors(ply_file):
    """分析PLY文件中的顶点颜色分布"""
    print(f"正在加载 {ply_file}...")
    
    # 加载网格
    mesh = o3d.io.read_triangle_mesh(ply_file)
    if not mesh.has_vertex_colors():
        print("错误: PLY文件不包含顶点颜色")
        return None, None
    
    # 获取颜色
    colors = np.asarray(mesh.vertex_colors)
    
    # 转换为RGB整数格式 (0-255)
    colors_int = np.round(colors * 255).astype(np.int32)
    
    # 创建颜色元组列表以便统计
    color_tuples = [tuple(c) for c in colors_int]
    
    # 统计不同颜色的数量
    color_counts = Counter(color_tuples)
    
    # 移除白色 (255,255,255)
    if (255, 255, 255) in color_counts:
        del color_counts[(255, 255, 255)]
        print("已移除白色 (255,255,255)")
    
    print(f"发现 {len(color_counts)} 种不同颜色")
    
    # 显示所有颜色
    print("\n所有颜色:")
    for i, (color, count) in enumerate(color_counts.most_common()):
        percentage = count / len(color_tuples) * 100
        print(f"{i+1}. RGB{color}: {count} 顶点 ({percentage:.2f}%)")
    
    return colors_int, color_counts

def extract_mesh_by_color(input_ply, target_color, output_file=None, tolerance=3):
    """
    提取PLY文件中指定颜色的顶点并组成新的网格
    参数:
        input_ply: 输入PLY文件路径
        target_color: 目标颜色 [r, g, b] (0-255)
        output_file: 输出文件路径
        tolerance: 颜色匹配容差 (0-255)
    返回:
        提取的网格
    """
    print(f"正在从 {input_ply} 提取颜色为 {target_color} 的网格...")
    
    # 加载网格
    mesh = o3d.io.read_triangle_mesh(input_ply)
    if not mesh.has_vertex_colors():
        print("错误: PLY文件不包含顶点颜色")
        return None
    
    # 将目标颜色归一化到0-1范围
    target_color_normalized = np.array(target_color) / 255.0
    
    # 获取顶点和颜色
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    triangles = np.asarray(mesh.triangles)
    
    # 找到匹配目标颜色的顶点索引
    # 考虑到浮点精度和颜色容差，我们使用欧氏距离
    color_diffs = np.sqrt(np.sum((colors - target_color_normalized)**2, axis=1))
    tolerance_normalized = tolerance / 255.0
    matching_vertices = np.where(color_diffs <= tolerance_normalized)[0]
    
    print(f"找到 {len(matching_vertices)} 个颜色匹配的顶点")
    
    if len(matching_vertices) == 0:
        print("没有找到匹配的顶点，尝试增加容差值")
        return None
    
    # 创建匹配顶点的集合，用于快速查找
    matching_vertices_set = set(matching_vertices)
    
    # 找到所有三个顶点都匹配目标颜色的面
    matching_triangles = []
    for i, triangle in enumerate(triangles):
        if (triangle[0] in matching_vertices_set and 
            triangle[1] in matching_vertices_set and 
            triangle[2] in matching_vertices_set):
            matching_triangles.append(i)
    
    print(f"找到 {len(matching_triangles)} 个匹配的面")
    
    if len(matching_triangles) == 0:
        print("没有找到匹配的面，无法创建网格")
        return None
    
    # 提取匹配的面
    extracted_triangles = triangles[matching_triangles]
    
    # 创建新顶点索引映射
    unique_vertices = set()
    for triangle in extracted_triangles:
        unique_vertices.update(triangle)
    
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
    
    # 创建新网格的顶点和面
    new_vertices = [vertices[i] for i in unique_vertices]
    new_colors = [colors[i] for i in unique_vertices]
    new_triangles = [[vertex_map[v] for v in triangle] for triangle in extracted_triangles]
    
    # 创建新的Open3D网格
    extracted_mesh = o3d.geometry.TriangleMesh()
    extracted_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    extracted_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    extracted_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    
    # 修复网格方向
    extracted_mesh.compute_vertex_normals()
    extracted_mesh.compute_triangle_normals()
    
    # 保存结果
    if output_file:
        o3d.io.write_triangle_mesh(output_file, extracted_mesh)
        print(f"已将提取的网格保存到 {output_file}")
    
    return extracted_mesh

def repair_mesh_to_watertight(mesh, output_file=None):
    """修复网格使其变为水密"""
    # 使用trimesh进行修复
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        o3d.io.write_triangle_mesh(tmp_file.name, mesh)
        trimesh_mesh = trimesh.load(tmp_file.name)
    os.unlink(tmp_file.name)
    
    # 使用voxelization确保水密性
    try:
        voxel_size = trimesh_mesh.bounding_box.extents.min() / 100
        voxel_mesh = trimesh_mesh.voxelized(pitch=voxel_size).marching_cubes
        
        if output_file:
            voxel_mesh.export(output_file)
            print(f"已将水密网格保存到 {output_file}")
            
        return True
    except Exception as e:
        print(f"修复网格为水密失败: {e}")
        return False

def extract_all_color_meshes(input_ply, output_dir="color_meshes", min_faces=100, tolerance=3):
    """提取所有颜色的网格"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    colors_int, color_counts = analyze_ply_colors(input_ply)
    
    # 只处理主要的几种颜色
    main_colors = [color for color, count in color_counts.most_common()]
    
    for i, color in enumerate(main_colors):
        color_name = f"{color[0]}_{color[1]}_{color[2]}"
        print(f"\n处理颜色 {i+1}/{len(main_colors)}: RGB{color}")
        
        # 提取该颜色的网格
        output_ply = os.path.join(output_dir, f"mesh_color_{color_name}.ply")
        mesh = extract_mesh_by_color(input_ply, color, output_ply, tolerance)
        
        if mesh is not None:
            # 检查面的数量
            triangles = np.asarray(mesh.triangles)
            if len(triangles) < min_faces:
                print(f"面数量({len(triangles)})低于阈值({min_faces})，跳过水密修复")
                continue
                
            # 修复为水密网格
            output_stl = os.path.join(output_dir, f"mesh_color_{color_name}_watertight.stl")
            repair_mesh_to_watertight(mesh, output_stl)

def visualize_mesh(mesh_file):
    """可视化网格"""
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], 
                                     window_name="网格可视化",
                                     width=800, height=600,
                                     mesh_show_back_face=True)

if __name__ == "__main__":
    input_ply = "data/midoutput-instseg.ply"
    
    if not os.path.exists(input_ply):
        print(f"错误: 文件 {input_ply} 不存在")
    else:
        # 分析颜色
        print("===== 分析颜色 =====")
        _, _ = analyze_ply_colors(input_ply)
        
        # 提取所有颜色的网格
        print("\n===== 提取所有主要颜色的网格 =====")
        extract_all_color_meshes(input_ply, output_dir="data/color_meshes")
        
        print("\n处理完成! 结果保存在 data/color_meshes 目录")
