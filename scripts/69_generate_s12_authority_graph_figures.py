#!/usr/bin/env python3
"""Generate authority, portfolio, graph, moving-load and field figures F01-F06."""

from __future__ import annotations

import importlib.util,json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[1];S10=ROOT/"s10_nested_grouped_oof";DATASET=S10/"S10_ORIGINAL_68CASE_DATASET.h5";GRAPH=ROOT.parent/"structure_preserving_pigno_v4"/"s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1"/"S8_GRAPH_INPUTS.npz"
_spec=importlib.util.spec_from_file_location("fig_utils",ROOT/"scripts"/"65_generate_s12_core_oof_figures.py");_fig=importlib.util.module_from_spec(_spec);assert _spec.loader is not None;_spec.loader.exec_module(_fig)


def setup3d(ax,coords):
    span=np.ptp(coords,axis=0)
    ax.set_box_aspect(np.maximum(span,1e-9))
    ax.set_proj_type("ortho")
    ax.view_init(elev=22,azim=-65,roll=0)
    ax.set_xlabel("X transverse (m)",fontsize=8,labelpad=5)
    ax.set_ylabel("Z longitudinal (m)",fontsize=8,labelpad=7)
    ax.set_zlabel("Y vertical (m)",fontsize=8,labelpad=5)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(axis="both",which="major",labelsize=6,pad=0)
    ax.grid(False)


def f01():
    stages=[("S0","Evidence"),("S1","FEM authority"),("S2","Graph/modal"),("S3","Sources"),("S4","6 routes frozen"),("S5","Oracle floors"),("S6","Capacity"),("S7","Repairs"),("S8","Factorial panel"),("S9","Multifidelity HPO"),("S10","Nested OOF"),("S11","5 seeds"),("S12","Diagnostics"),("S13","New FEM panel\nnot authorized"),("S14","Decision/package")];rows=[];fig,ax=plt.subplots(figsize=(14,4.5));ax.axis("off")
    for i,(stage,label) in enumerate(stages):
        row=i//8;col=i%8;x=col*1.7 if row==0 else (7-col)*1.7;y=2.5-row*2;face="#E8F0F8" if stage not in {"S13","S14"} else "#F3EEE4";ax.text(x,y,f"{stage}\n{label}",ha="center",va="center",bbox={"boxstyle":"round,pad=.35","facecolor":face,"edgecolor":"#555555"},fontsize=8);rows.append({"order":i,"stage":stage,"label":label,"x":x,"y":y})
        if i>0:
            prev=rows[-2];ax.annotate("",xy=(x,y),xytext=(prev["x"],prev["y"]),arrowprops={"arrowstyle":"->","color":"#555555","lw":1})
    ax.set_xlim(-1,13);ax.set_ylim(-.5,3.5);ax.set_title("Frozen physics-informed operator portfolio workflow");fig.tight_layout();_fig.save(fig,"F01","Frozen physics-informed operator portfolio workflow","State-machine view of the finite six-family campaign. S13 is design-only because a genuinely new FEM panel requires explicit authorization.",pd.DataFrame(rows),{"units":"not applicable","quantity":"campaign stages"})


def f02():
    families=pd.read_csv(ROOT/"PORTFOLIO_FAMILY_MATRIX.csv");mechanisms={"Beam graph":"graph","spectral/TFNO":"tfno|fourier","modal":"modal","weak/Galerkin":"galerkin|virtual work|weak","energy/passivity":"hamilton|energy|passiv","local frames":"local frame|rotation|polar|axial","load-dependent ROM":"ritz|krylov|soar|load-conditioned","multioperator":"separate|specialized|q/v/a"};rows=[]
    for _,row in families.iterrows():
        text=" ".join(str(value).lower() for value in row.values)
        for mechanism,patterns in mechanisms.items():rows.append({"route":row.route_id,"mechanism":mechanism,"present":int(any(pattern in text for pattern in patterns.split("|")))})
    frame=pd.DataFrame(rows);pivot=frame.pivot(index="route",columns="mechanism",values="present");fig,ax=plt.subplots(figsize=(10,4));image=ax.imshow(pivot.to_numpy(),cmap=matplotlib.colors.ListedColormap(["#F0F0F0","#2463A6"]),vmin=0,vmax=1,aspect="auto");ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=35,ha="right");ax.set_yticks(range(len(pivot.index)),pivot.index);ax.grid(False);ax.set_title("Distinct physics and representation mechanisms across six routes")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):ax.text(j,i,"●" if pivot.iloc[i,j] else "—",ha="center",va="center",color="white" if pivot.iloc[i,j] else "#777777")
    fig.tight_layout();_fig.save(fig,"F02","Distinct physics and representation mechanisms across six routes","Binary mechanism map derived from the frozen family matrix. It visualizes nonredundancy; it is not a performance ranking.",frame,{"units":"binary presence","quantity":"portfolio architecture"})


def f03(graph):
    coords=graph["graph_coords_m"][:,[0,2,1]];edges=graph["edge_index"].T;segments=np.stack([coords[edges[:,0]],coords[edges[:,1]]],axis=1);fixed=np.any(graph["fixed_dof"][:,:3],axis=1);obs=graph["observation_to_graph"]
    fig=plt.figure(figsize=(14,7));ax=fig.add_subplot(111,projection="3d");ax.add_collection3d(Line3DCollection(segments,colors="#7A7A7A",linewidths=.15,alpha=.28));ax.scatter(coords[obs,0],coords[obs,1],coords[obs,2],s=3,color="#2463A6",label="512 observations");ax.scatter(coords[fixed,0],coords[fixed,1],coords[fixed,2],s=8,color="#A34A3A",label="translation-restrained nodes");setup3d(ax,coords);ax.set_title("Active FEM Beam graph and observation operator — true aspect",pad=12);ax.legend(fontsize=7,loc="upper right");fig.subplots_adjust(left=.04,right=.96,bottom=.10,top=.91)
    edge_frame=pd.DataFrame({"kind":"edge","edge_index":np.arange(len(edges)),"node_i":edges[:,0],"node_j":edges[:,1],"X_i_m":coords[edges[:,0],0],"Z_i_m":coords[edges[:,0],1],"Y_i_m":coords[edges[:,0],2],"X_j_m":coords[edges[:,1],0],"Z_j_m":coords[edges[:,1],1],"Y_j_m":coords[edges[:,1],2]});_fig.save(fig,"F03","Active FEM Beam graph and observation operator — true aspect","All 48,430 active Beam edges, 512 observation nodes and translation-restrained graph nodes. Geometry is plotted X-transverse, Z-longitudinal, Y-vertical with true aspect.",edge_frame,{"units":"m","quantity":"active Beam graph","node_count":int(len(coords)),"edge_count":int(len(edges)),"observation_count":int(len(obs))})


def f04(graph):
    coords=graph["graph_coords_m"];edges=graph["edge_index"].T;frames=graph["edge_local_frame_R_local_from_global"];sample=np.linspace(0,len(edges)-1,120,dtype=int);mid=.5*(coords[edges[sample,0]]+coords[edges[sample,1]]);length=np.linalg.norm(coords[edges[sample,1]]-coords[edges[sample,0]],axis=1);scale=np.median(length)*2;colors=["#A34A3A","#2463A6","#D18B20"];labels=["local x","local y","local z"]
    fig,panels=plt.subplots(1,3,figsize=(16,5.3));projections=[("Plan",2,0,"Z longitudinal (m)","X transverse (m)",2.0),("Elevation",2,1,"Z longitudinal (m)","Y vertical (m)",2.0),("Cross-section",0,1,"X transverse (m)","Y vertical (m)",1.2)]
    rows=[]
    for local_axis,(color,label) in enumerate(zip(colors,labels)):
        vectors=frames[sample,local_axis,:]
        for edge_index,point,vector in zip(sample,mid,vectors):rows.append({"edge_index":int(edge_index),"local_axis":label,"mid_X_m":point[0],"mid_Y_m":point[1],"mid_Z_m":point[2],"dir_X":vector[0],"dir_Y":vector[1],"dir_Z":vector[2]})
    for panel,(title,horizontal,vertical,xlabel,ylabel,glyph_length) in zip(panels,projections):
        panel.scatter(mid[:,horizontal],mid[:,vertical],s=2,color="#B8B8B8",alpha=.45,zorder=1)
        for local_axis,color in enumerate(colors):
            vector=frames[sample,local_axis,:]
            start=mid[:,[horizontal,vertical]]
            end=start+glyph_length*vector[:,[horizontal,vertical]]
            panel.add_collection(LineCollection(np.stack([start,end],axis=1),colors=color,linewidths=.65,alpha=.8,zorder=2))
        panel.set_title(title);panel.set_xlabel(xlabel);panel.set_ylabel(ylabel);panel.set_aspect("equal",adjustable="box");panel.autoscale();panel.margins(.04)
    handles=[Line2D([0],[0],color=color,lw=1.3,label=label) for color,label in zip(colors,labels)]
    panels[-1].legend(handles=handles,fontsize=7,loc="upper right");fig.suptitle("Audited local Beam frames — three true-aspect projections");fig.tight_layout();_fig.save(fig,"F04","Audited local Beam frames — three true-aspect projections","Projected local x/y/z directions for 120 deterministic edge samples in plan, elevation and cross-section. Each projection preserves the physical aspect of its two displayed axes; glyph lengths are visual only and the source direction cosines remain unitless and unscaled.",pd.DataFrame(rows),{"units":"m for positions; unitless direction cosines","quantity":"local Beam frames","sample_edge_count":120,"glyph_length_m":{"plan":2.0,"elevation":2.0,"cross_section":1.2}})


def f05(fem):
    cases=[x.decode() if isinstance(x,bytes) else str(x) for x in fem["case_id"][:]];static=fem["causal/static_features"][:];candidates=np.where((static[:,1]==2)&(static[:,0]==52))[0];index=int(candidates[-1]);case=cases[index];time=fem["time_s"][:];positions=fem["causal/axle_position_m"][index];active=fem["causal/track_active"][index].astype(bool);bridge=float(static[index,10]);rows=[];fig,ax=plt.subplots(figsize=(10,5))
    for track in range(2):
        if not active[track]:continue
        for axle in range(22):
            position=positions[:,track,axle];mask=(position>=0)&(position<=bridge);ax.plot(time[mask],position[mask],color="#2463A6" if track==0 else "#D18B20",linewidth=.45,alpha=.8)
            rows.extend({"case_id":case,"time_s":float(t),"track":track+1,"axle":axle+1,"position_Z_m":float(z),"inside_bridge":bool(inside)} for t,z,inside in zip(time,position,mask))
    ax.set_xlabel("Time (s)");ax.set_ylabel("Longitudinal axle position Z (m)");ax.set_ylim(0,bridge);ax.set_title(f"Causal axle trajectories on active rails — {case}");ax.text(.99,.02,f"22 axles/train; active tracks={np.where(active)[0].tolist()}",transform=ax.transAxes,ha="right",va="bottom",fontsize=8);fig.tight_layout();_fig.save(fig,"F05","Causal axle trajectories on active rails",f"Axle position histories for {case}. Only positions inside the {bridge:g} m bridge span are drawn; source data retain entry and exit evolution.",pd.DataFrame(rows),{"units":"s and m","quantity":"moving-load kinematics","case_id":case,"bridge_length_m":bridge})


def f06(fem):
    cases=[x.decode() if isinstance(x,bytes) else str(x) for x in fem["case_id"][:]];case="V52_CPLUS_E8_C12_2T" if "V52_CPLUS_E8_C12_2T" in cases else cases[-1];index=cases.index(case);time=fem["time_s"][:];time_index=int(np.argmin(np.abs(time-9.025)));coords=fem["observation/coords_m"][:];total=fem["response/total_translation_m"][index,time_index,:,1];delta=fem["response/delta_translation_m"][index,time_index,:,1];base=total-delta;fields=[("base",base),("incremental",delta),("total",total)];rows=[];fig,axes=plt.subplots(3,2,figsize=(13,8.5));limits=max(float(np.max(np.abs(np.concatenate([base,delta,total])))),1e-12)
    for row_index,(name,values) in enumerate(fields):
        plan,elevation=axes[row_index]
        points=plan.scatter(coords[:,2],coords[:,0],c=1000*values,cmap="coolwarm",vmin=-1000*limits,vmax=1000*limits,s=8)
        elevation.scatter(coords[:,2],coords[:,1],c=1000*values,cmap="coolwarm",vmin=-1000*limits,vmax=1000*limits,s=8)
        plan.set_title(f"{name.capitalize()} — plan");plan.set_xlabel("Z longitudinal (m)");plan.set_ylabel("X transverse (m)");plan.set_aspect("equal",adjustable="box")
        elevation.set_title(f"{name.capitalize()} — elevation");elevation.set_xlabel("Z longitudinal (m)");elevation.set_ylabel("Y vertical (m)");elevation.set_aspect("equal",adjustable="box")
        fig.colorbar(points,ax=[plan,elevation],fraction=.012,pad=.015,label="Y displacement (mm)")
        rows.extend({"case_id":case,"time_s":float(time[time_index]),"view":name,"observation_node_zero_based":node,"X_m":coords[node,0],"Y_m":coords[node,1],"Z_m":coords[node,2],"Y_displacement_m":values[node]} for node in range(512))
    fig.suptitle(f"FEM/COMSOL base, incremental and total vertical fields — {case}, t={time[time_index]:.3f} s",y=.995);fig.subplots_adjust(left=.07,right=.91,bottom=.06,top=.95,hspace=.48,wspace=.22);_fig.save(fig,"F06","FEM/COMSOL base, incremental and total vertical fields",f"Plan and elevation projections of reference fields at the saved time nearest 9.025 s for {case}. Each projection preserves its physical aspect, and one symmetric color scale is common to base, incremental and total fields.",pd.DataFrame(rows),{"units":"m in source; mm in color scale","quantity":"Y vertical displacement","case_id":case,"time_s":float(time[time_index]),"geometry":"true-aspect plan and elevation projections"})


def main():
    if (_fig.FIGURES/"F01.png").exists():raise FileExistsError("Authority/graph figures already exist")
    _fig.style();f01();f02()
    graph=np.load(GRAPH,allow_pickle=True);f03(graph);f04(graph)
    with h5py.File(DATASET,"r") as fem:f05(fem);f06(fem)
    report={"status":"PASS_S12_AUTHORITY_GRAPH_FIGURES","figure_ids":["F01","F02","F03","F04","F05","F06"],"training_or_tuning_performed":False};_fig.atomic_json(ROOT/"s12_final_diagnostics"/"S12_AUTHORITY_GRAPH_FIGURES_REPORT.json",report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
