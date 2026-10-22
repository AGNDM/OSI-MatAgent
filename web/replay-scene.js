import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const number = value => typeof value === 'number' && Number.isFinite(value);
const mix = (a,b,t) => number(a)&&number(b)?a+(b-a)*t:b;

export class InstrumentScene {
  constructor(container, status) {
    this.container=container; this.status=status; this.dirty=true; this.ready=false;
    this.rotorAngle=0;this.lastFrameTime=null;
    this.scene=new THREE.Scene();
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.6));
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;
    this.renderer.toneMapping=THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure=1.0;
    container.appendChild(this.renderer.domElement);
    this.camera=new THREE.PerspectiveCamera(38,1,0.01,30);
    this.camera.up.set(0,0,1);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);
    this.controls.enableDamping=true;this.controls.dampingFactor=.12;
    this.controls.minDistance=.35;this.controls.maxDistance=5;
    this.controls.maxPolarAngle=Math.PI*.85;
    this.controls.addEventListener('change',()=>this.dirty=true);
    this.resetCamera();
    this.scene.add(new THREE.HemisphereLight(0xe1f4ed,0x55615f,1.2));
    const key=new THREE.DirectionalLight(0xfff7e6,2.0);key.position.set(1,-3,3);this.scene.add(key);
    const rim=new THREE.DirectionalLight(0xa2dbe2,1.2);rim.position.set(-2,1,2);this.scene.add(rim);
    const fill=new THREE.DirectionalLight(0xffffff,.6);fill.position.set(-2,-2,1);this.scene.add(fill);
    const floor=new THREE.Mesh(new THREE.CircleGeometry(1.5,64),new THREE.MeshStandardMaterial({color:0x263d38,roughness:1,metalness:0,transparent:true,opacity:.65}));floor.position.z=-.006;this.scene.add(floor);
    const grid=new THREE.GridHelper(3,30,0x4d7064,0x3b514a);grid.rotation.x=Math.PI/2;grid.position.z=-.004;grid.material.transparent=true;grid.material.opacity=.32;this.scene.add(grid);
    this.resizeObserver=new ResizeObserver(()=>this.resize());this.resizeObserver.observe(container);this.resize();
    this.frame=this.frame.bind(this);requestAnimationFrame(this.frame);
  }
  resize(){const r=this.container.getBoundingClientRect();if(!r.width||!r.height)return;this.renderer.setSize(r.width,r.height,false);this.camera.aspect=r.width/r.height;this.camera.updateProjectionMatrix();this.dirty=true;}
  resetCamera(){this.camera.position.set(1.25,-1.85,1.23);this.controls.target.set(.05,-.05,.57);this.controls.update();this.dirty=true;}
  focusReactor(){this.camera.position.set(.63,-1.20,.88);this.controls.target.set(.128,-.26,.49);this.controls.update();this.dirty=true;}
  async load(url){
    this.status.hidden=false;this.status.textContent='Loading RC1mx + HFCal model…';
    try{
      const gltf=await new GLTFLoader().loadAsync(url);
      this.model=gltf.scene;this.scene.add(this.model);this.parts={};
      this.model.traverse(node=>{
        const name=node.name.split('__').pop();this.parts[name]=node;
        if(node.isMesh){
          node.material=node.material.clone();
          node.material.roughness=Math.max(.3,node.material.roughness??.45);
          node.material.metalness=Math.min(.5,node.material.metalness??0);
          if(['reactor_jacket','reactor_vessel','reactor_cover'].includes(name)){
            node.material.transparent=true;node.material.opacity=name==='reactor_cover'?.30:.18;
            node.material.depthWrite=false;node.material.side=THREE.DoubleSide;
            node.renderOrder=3;
          }
          if(['stirrer_shaft','stirrer_impeller'].includes(name)){
            node.material.color.setHex(0xef932e);node.material.metalness=.18;node.material.roughness=.45;
            node.material.transparent=false;node.material.opacity=1;node.material.depthWrite=true;
            node.material.emissive.setHex(0x8d410d);node.material.emissiveIntensity=.12;
          }
        }
      });
      this.heatParts={};
      for(const name of ['oil_heater','reactor_jacket','cal_heater']){
        const part=this.parts[name];if(!part?.material)continue;
        this.heatParts[name]={material:part.material,color:part.material.color.clone(),emissive:part.material.emissive.clone(),intensity:part.material.emissiveIntensity};
      }
      this.heatColor=new THREE.Color(0xff6228);
      this.displays={};
      for(const name of ['tr_digits','tj_digits','speed_digits'])this.makeDisplay(name);
      this.ready=true;this.setSupported(this.supported??true);
      if(this.target)this.applyState(this.target);
      this.dirty=true;
    }catch(error){this.status.textContent='3D model unavailable. Run tools/fetch_replay_assets.py, then refresh. '+error.message;throw error;}
  }
  setSupported(supported){if(this.model)this.model.visible=supported;this.supported=supported;this.status.hidden=supported&&this.ready;if(!supported)this.status.textContent='No 3D model is configured for this instrument. Recorded messages and states remain available.';this.dirty=true;}
  makeDisplay(name){
    const original=this.parts[name];if(!original)return;
    const box=new THREE.Box3().setFromObject(original),size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3());
    const canvas=document.createElement('canvas');canvas.width=320;canvas.height=112;
    const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
    const plane=new THREE.Mesh(new THREE.PlaneGeometry(Math.max(size.x,.045),Math.max(size.z,.018)),new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide,toneMapped:false}));
    plane.rotation.x=Math.PI/2;plane.position.set(center.x,box.min.y-.0015,center.z);
    this.model.add(plane);original.visible=false;
    this.displays[name]={canvas,texture,plane,text:null};
  }
  writeDisplay(name,value,decimals){const d=this.displays?.[name];if(!d)return;const text=number(value)?value.toFixed(decimals):'—';if(text===d.text)return;d.text=text;const ctx=d.canvas.getContext('2d');ctx.fillStyle='#09110d';ctx.fillRect(0,0,320,112);ctx.font='72px monospace';ctx.textAlign='right';ctx.fillStyle='#ffba65';ctx.fillText(text,308,84);d.texture.needsUpdate=true;}
  applyState(frame){
    if(!this.ready)return;
    this.writeDisplay('tr_digits',frame.reactor_c,1);this.writeDisplay('tj_digits',frame.jacket_c,1);this.writeDisplay('speed_digits',frame.rpm,0);
    const setLamp=(name,on,color)=>{const p=this.parts[name];if(!p?.material)return;p.material.emissive.setHex(color);p.material.emissiveIntensity=on===true?1.5:0;p.material.color.setHex(on===true?color:0x24302a);};
    setLamp('lamp_status_ok',frame.power===true&&frame.emergency!==true,0x68d49c);
    setLamp('lamp_status_alarm',frame.emergency,0xe77a57);
    const glow=(name,amount,strength)=>{const part=this.heatParts?.[name];if(!part)return;const level=number(amount)?Math.max(0,Math.min(1,amount)):0;part.material.color.copy(part.color).lerp(this.heatColor,level*.5);part.material.emissive.copy(part.emissive).lerp(this.heatColor,level);part.material.emissiveIntensity=part.intensity+level*strength;};
    glow('oil_heater',frame.heating_level,3);
    glow('reactor_jacket',frame.heating_level,1.5);
    glow('cal_heater',frame.cal_heating_level,3);
    if(this.parts.stirrer_shaft)this.parts.stirrer_shaft.rotation.z=this.rotorAngle;
    if(this.parts.drain_valve_handle&&number(frame.drain_angle))this.parts.drain_valve_handle.rotation.y=frame.drain_angle;
    this.dirty=true;
  }
  transition(target,duration=900){
    const previous=this.current||target;
    // The rotor has its own visual phase, independent of the historical clock.
    this.target={...target};
    if(matchMedia('(prefers-reduced-motion: reduce)').matches)duration=0;
    this.animation=duration?{from:{...previous},to:this.target,start:performance.now(),duration}:null;
    if(!duration){this.current=this.target;this.applyState(this.target);}
    this.dirty=true;
  }
  frame(now){
    requestAnimationFrame(this.frame);
    if(document.hidden){this.lastFrameTime=null;return;}
    const elapsed=this.lastFrameTime===null?0:Math.max(0,Math.min(.1,(now-this.lastFrameTime)/1000));
    this.lastFrameTime=now;
    this.controls.update();
    if(this.animation){const a=this.animation,t=Math.min(1,(now-a.start)/a.duration),ease=t*t*(3-2*t);this.current={...a.to};for(const k of ['reactor_c','jacket_c','rpm','drain_angle','heating_level','cal_heating_level'])this.current[k]=mix(a.from[k],a.to[k],ease);this.applyState(this.current);if(t===1)this.animation=null;}
    const rpm=this.current?.rpm;
    if(this.ready&&this.supported!==false&&number(rpm)&&rpm>0&&this.parts.stirrer_shaft){
      // RPM -> radians per wall-clock second. Pausing steps does not stop the rotor.
      this.rotorAngle=(this.rotorAngle+rpm*2*Math.PI/60*elapsed)%(2*Math.PI);
      this.parts.stirrer_shaft.rotation.z=this.rotorAngle;
      this.dirty=true;
    }
    if(this.dirty){this.renderer.render(this.scene,this.camera);this.dirty=false;}
  }
}
