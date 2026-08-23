(() => {

  // ============================================================
  // V12 MediaPipe Face Landmarker
  // ============================================================
  const MP_VERSION = "1.0.1";
  const MP_MODULE_URL =
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
  const MP_WASM_ROOT = `${MP_MODULE_URL}/wasm`;
  const MP_FACE_MODEL =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

  const MP_FACE_OVAL = [
    10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,
    400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,
    103,67,109
  ];
  const MP_LEFT_EYE = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466];
  const MP_RIGHT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246];
  const MP_LEFT_BROW = [336,296,334,293,300,285,295,282,283,276];
  const MP_RIGHT_BROW = [70,63,105,66,107,55,65,52,53,46];
  const MP_LIPS = [
    61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,
    87,178,88,95,78,191,80,81,82,13,312,311,310,415
  ];
  const MP_NOSE = [168,6,197,195,5,4,1,19,94,2,98,97,326,327,294,278,344,440];

  let FaceLandmarker = null;
  let ImageSegmenter = null;
  let FilesetResolver = null;
  let mpModulePromise = null;
  let mpFaceLandmarker = null;
  let mpFacePromise = null;
  let mpFaceReady = false;
  let mpFaceFailed = false;

  const MP_SEGMENT_MODEL =
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite";
  let mpImageSegmenter = null;
  let mpSegmenterPromise = null;

  function setFaceModelStatus(kind,text){
    const box=document.getElementById("faceModelStatus");
    const label=document.getElementById("faceModelStatusText");
    if(box){
      box.classList.remove("loading","ready","error","fallback");
      box.classList.add(kind);
    }
    if(label) label.textContent=text;
  }

  function withTimeout(promise,ms,label){
    let timer;
    const timeout=new Promise((_,reject)=>{
      timer=setTimeout(
        ()=>reject(new Error(`${label}逾時（${Math.round(ms/1000)} 秒）`)),
        ms
      );
    });
    return Promise.race([promise,timeout]).finally(()=>clearTimeout(timer));
  }

  async function loadMediaPipeModule(){
    if(FaceLandmarker && FilesetResolver && ImageSegmenter){
      return {FaceLandmarker,FilesetResolver,ImageSegmenter};
    }

    if(!mpModulePromise){
      mpModulePromise=(async()=>{
        setFaceModelStatus("loading",`載入 MediaPipe ${MP_VERSION}…`);

        // 使用 dynamic import，讓 CDN import 錯誤能被 try/catch 捕捉，
        // 而不是讓整個 app.js 在模組解析階段直接停止。
        const mod=await withTimeout(
          import(MP_MODULE_URL),
          20000,
          "MediaPipe JavaScript"
        );

        const vision=mod.default || mod;
        FaceLandmarker=mod.FaceLandmarker || vision.FaceLandmarker;
        ImageSegmenter=mod.ImageSegmenter || vision.ImageSegmenter;
        FilesetResolver=mod.FilesetResolver || vision.FilesetResolver;

        if(!FaceLandmarker || !FilesetResolver){
          throw new Error(
            "MediaPipe 模組已下載，但找不到 FaceLandmarker / FilesetResolver"
          );
        }

        return {FaceLandmarker,FilesetResolver,ImageSegmenter};
      })().catch(err=>{
        mpModulePromise=null;
        throw err;
      });
    }

    return mpModulePromise;
  }

  async function createMediaPipeFaceLandmarker(delegate){
    await loadMediaPipeModule();

    setFaceModelStatus(
      "loading",
      delegate==="GPU" ? "載入 WASM / 模型（GPU）…" : "載入 WASM / 模型（CPU）…"
    );

    const fileset = await withTimeout(
      FilesetResolver.forVisionTasks(MP_WASM_ROOT),
      20000,
      "MediaPipe WASM"
    );

    return await withTimeout(
      FaceLandmarker.createFromOptions(fileset,{
        baseOptions:{
          modelAssetPath:MP_FACE_MODEL,
          ...(delegate ? {delegate} : {})
        },
        runningMode:"IMAGE",
        numFaces:3,
        minFaceDetectionConfidence:.55,
        minFacePresenceConfidence:.55,
        minTrackingConfidence:.5,
        outputFaceBlendshapes:false,
        outputFacialTransformationMatrixes:false
      }),
      30000,
      "Face Landmarker 模型"
    );
  }

  async function initMediaPipeFaceLandmarker(){
    if(mpFaceLandmarker) return mpFaceLandmarker;
    if(mpFacePromise) return mpFacePromise;

    mpFacePromise=(async()=>{
      setFaceModelStatus("loading","正在連線 MediaPipe…");
      try{
        await loadMediaPipeModule();

        try{
          mpFaceLandmarker=await createMediaPipeFaceLandmarker("GPU");
        }catch(gpuError){
          console.warn("MediaPipe GPU 初始化失敗，改用 CPU/WASM。",gpuError);
          mpFaceLandmarker=await createMediaPipeFaceLandmarker(null);
        }

        mpFaceReady=true;
        mpFaceFailed=false;
        setFaceModelStatus("ready",`MediaPipe ${MP_VERSION} 已就緒`);
        return mpFaceLandmarker;
      }catch(err){
        console.error("MediaPipe Face Landmarker 初始化失敗",err);
        mpFaceReady=false;
        mpFaceFailed=true;
        mpFacePromise=null;
        setFaceModelStatus(
          "error",
          `MediaPipe 載入失敗：${err?.message || "未知錯誤"}`
        );
        throw err;
      }
    })();

    return mpFacePromise;
  }

  async function createMediaPipeImageSegmenter(delegate){
    await loadMediaPipeModule();
    if(!ImageSegmenter) throw new Error("目前 MediaPipe 模組沒有提供 ImageSegmenter");

    const fileset=await withTimeout(
      FilesetResolver.forVisionTasks(MP_WASM_ROOT),
      20000,
      "MediaPipe ImageSegmenter WASM"
    );

    return await withTimeout(
      ImageSegmenter.createFromOptions(fileset,{
        baseOptions:{
          modelAssetPath:MP_SEGMENT_MODEL,
          ...(delegate ? {delegate} : {})
        },
        runningMode:"IMAGE",
        outputCategoryMask:true,
        outputConfidenceMasks:false
      }),
      30000,
      "人物去背模型"
    );
  }

  async function initMediaPipeImageSegmenter(){
    if(mpImageSegmenter) return mpImageSegmenter;
    if(mpSegmenterPromise) return mpSegmenterPromise;

    mpSegmenterPromise=(async()=>{
      await loadMediaPipeModule();
      try{
        try{
          mpImageSegmenter=await createMediaPipeImageSegmenter("GPU");
        }catch(gpuError){
          console.warn("ImageSegmenter GPU 初始化失敗，改用 CPU/WASM。",gpuError);
          mpImageSegmenter=await createMediaPipeImageSegmenter(null);
        }
        return mpImageSegmenter;
      }catch(err){
        mpSegmenterPromise=null;
        throw err;
      }
    })();

    return mpSegmenterPromise;
  }

  async function runImageSegmentation(inputCanvas){
    const segmenter=await initMediaPipeImageSegmenter();

    const task=new Promise((resolve,reject)=>{
      let settled=false;
      const done=result=>{
        if(settled || !result) return;
        settled=true;
        resolve(result);
      };

      try{
        const result=segmenter.segment(inputCanvas,done);
        if(result && typeof result.then==="function"){
          result.then(done).catch(reject);
        }else if(result && (result.categoryMask || result.confidenceMasks)){
          done(result);
        }
      }catch(err){
        reject(err);
      }
    });

    return await withTimeout(task,30000,"人物去背分析");
  }

  function buildForegroundMaskCanvas(segmentResult,targetW,targetH,featherPx=1.5){
    const categoryMask=segmentResult?.categoryMask;
    if(!categoryMask) throw new Error("去背模型沒有回傳 categoryMask");

    const mw=categoryMask.width;
    const mh=categoryMask.height;
    const categories=categoryMask.getAsUint8Array();

    const small=document.createElement("canvas");
    small.width=mw;
    small.height=mh;
    const sctx2=small.getContext("2d",{willReadFrequently:true});
    const maskImage=sctx2.createImageData(mw,mh);

    for(let i=0;i<categories.length;i++){
      const foreground=categories[i]===0 ? 0 : 255;
      const p=i*4;
      maskImage.data[p]=255;
      maskImage.data[p+1]=255;
      maskImage.data[p+2]=255;
      maskImage.data[p+3]=foreground;
    }
    sctx2.putImageData(maskImage,0,0);

    const full=document.createElement("canvas");
    full.width=targetW;
    full.height=targetH;
    const fctx=full.getContext("2d");
    fctx.clearRect(0,0,targetW,targetH);
    fctx.imageSmoothingEnabled=true;
    fctx.imageSmoothingQuality="high";
    if(featherPx>0) fctx.filter=`blur(${featherPx}px)`;
    fctx.drawImage(small,0,0,targetW,targetH);
    fctx.filter="none";

    if(typeof categoryMask.close==="function"){
      try{categoryMask.close();}catch{}
    }
    return full;
  }

  async function removePersonBackground(outputMode="white",featherPx=1.5){
    if(!source.width || !source.height) return;

    $("removeBgBtn").disabled=true;
    $("bgRemoveInfo").textContent="正在載入人物分割模型並分析背景…";

    try{
      const result=await runImageSegmentation(source);
      const mask=buildForegroundMaskCanvas(
        result,source.width,source.height,featherPx
      );

      const out=document.createElement("canvas");
      out.width=source.width;
      out.height=source.height;
      const ctx=out.getContext("2d");

      ctx.clearRect(0,0,out.width,out.height);
      ctx.drawImage(source,0,0);
      ctx.globalCompositeOperation="destination-in";
      ctx.drawImage(mask,0,0);
      ctx.globalCompositeOperation="source-over";

      if(outputMode==="white"){
        ctx.globalCompositeOperation="destination-over";
        ctx.fillStyle="#ffffff";
        ctx.fillRect(0,0,out.width,out.height);
        ctx.globalCompositeOperation="source-over";
      }

      source.width=out.width;
      source.height=out.height;
      sctx.clearRect(0,0,source.width,source.height);
      sctx.drawImage(out,0,0);

      sourceHasTransparency=(outputMode==="transparent");
      sourceDirty=true;
      smartSpots=[];
      smartFaceRegion=null;
      $("smartApplyBtn").disabled=true;
      pushHistory();
      touchCurrentBatchItem();
      updateMeta();
      await renderPreview();

      $("bgRemoveInfo").textContent=outputMode==="transparent"
        ? "去背完成：背景目前為透明。請使用 PNG 下載；JPG 會自動轉為白色背景。"
        : "去背完成：背景已填成白色，可使用 JPG 或 PNG 下載。";
    }catch(err){
      console.error("人物去背失敗",err);
      $("bgRemoveInfo").textContent=
        "人物去背失敗：" + (err?.message || "未知錯誤") +
        "。其他修圖功能不受影響，可稍後重試。";
    }finally{
      $("removeBgBtn").disabled=!source.width;
    }
  }

  function mpLandmarkPixels(landmarks,canvas){
    return landmarks.map(p=>({
      x:p.x*canvas.width,
      y:p.y*canvas.height,
      z:p.z || 0
    }));
  }

  function mpBounds(points,indices){
    let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
    for(const idx of indices){
      const p=points[idx];
      if(!p) continue;
      minX=Math.min(minX,p.x);
      minY=Math.min(minY,p.y);
      maxX=Math.max(maxX,p.x);
      maxY=Math.max(maxY,p.y);
    }
    if(!Number.isFinite(minX)) return null;
    return {x:minX,y:minY,w:maxX-minX,h:maxY-minY};
  }

  function mpLargestFaceResult(result,canvas){
    const faces=result?.faceLandmarks || [];
    if(!faces.length) return null;

    let best=null;
    for(const landmarks of faces){
      const points=mpLandmarkPixels(landmarks,canvas);
      const bbox=mpBounds(points,MP_FACE_OVAL);
      if(!bbox) continue;
      const area=bbox.w*bbox.h;
      if(!best || area>best.area){
        best={landmarks,points,bbox,area};
      }
    }
    return best;
  }

  async function detectMediaPipeFace(inputCanvas){
    if(!inputCanvas?.width || !inputCanvas?.height) return null;
    try{
      const landmarker=await initMediaPipeFaceLandmarker();
      const result=landmarker.detect(inputCanvas);
      return mpLargestFaceResult(result,inputCanvas);
    }catch(err){
      return null;
    }
  }

  function mpAveragePoint(points,indices){
    let x=0,y=0,n=0;
    for(const idx of indices){
      const p=points[idx];
      if(!p) continue;
      x+=p.x; y+=p.y; n++;
    }
    return n ? {x:x/n,y:y/n} : {x:0,y:0};
  }

  function mpPolygon(points,indices){
    return indices.map(i=>points[i]).filter(Boolean);
  }

  function mpPointInPolygon(x,y,poly){
    let inside=false;
    for(let i=0,j=poly.length-1;i<poly.length;j=i++){
      const xi=poly[i].x, yi=poly[i].y;
      const xj=poly[j].x, yj=poly[j].y;
      const intersect=((yi>y)!==(yj>y)) &&
        (x < (xj-xi)*(y-yi)/((yj-yi)||1e-9)+xi);
      if(intersect) inside=!inside;
    }
    return inside;
  }

  function mpShrinkPolygon(poly,factor){
    if(!poly.length) return [];
    const cx=poly.reduce((s,p)=>s+p.x,0)/poly.length;
    const cy=poly.reduce((s,p)=>s+p.y,0)/poly.length;
    return poly.map(p=>({
      x:cx+(p.x-cx)*factor,
      y:cy+(p.y-cy)*factor
    }));
  }

  function mpExpandedBounds(points,indices,padX,padY){
    const b=mpBounds(points,indices);
    if(!b) return null;
    return {
      x:b.x-b.w*padX,
      y:b.y-b.h*padY,
      w:b.w*(1+padX*2),
      h:b.h*(1+padY*2)
    };
  }

  function mpPointInRect(x,y,r){
    return !!r && x>=r.x && x<=r.x+r.w && y>=r.y && y<=r.y+r.h;
  }

  function mpFaceOverlayRegion(face){
    if(!face) return null;
    return {
      x:face.bbox.x,
      y:face.bbox.y,
      w:face.bbox.w,
      h:face.bbox.h,
      landmarks:face.points
    };
  }

  function mpEstimateBackgroundColor(canvas){
    const ctx=canvas.getContext("2d",{willReadFrequently:true});
    const w=canvas.width,h=canvas.height;
    const size=Math.max(3,Math.round(Math.min(w,h)*.035));
    const patches=[
      [0,0],[Math.max(0,w-size),0],
      [0,Math.max(0,h-size)],[Math.max(0,w-size),Math.max(0,h-size)]
    ];
    const colors=[];
    for(const [x,y] of patches){
      const data=ctx.getImageData(x,y,Math.min(size,w-x),Math.min(size,h-y)).data;
      let r=0,g=0,b=0,n=0;
      for(let i=0;i<data.length;i+=4){
        if(data[i+3]<32) continue;
        r+=data[i]; g+=data[i+1]; b+=data[i+2]; n++;
      }
      if(n) colors.push([r/n,g/n,b/n]);
    }
    if(!colors.length) return [255,255,255];
    colors.sort((a,b)=>(a[0]+a[1]+a[2])-(b[0]+b[1]+b[2]));
    return colors[Math.floor(colors.length/2)];
  }

  function mpFindHeadTop(canvas,face){
    const ctx=canvas.getContext("2d",{willReadFrequently:true});
    const data=ctx.getImageData(0,0,canvas.width,canvas.height).data;
    const bg=mpEstimateBackgroundColor(canvas);
    const faceTop=face.points[10]?.y ?? face.bbox.y;
    const centerX=(face.bbox.x+face.bbox.w/2);
    const left=Math.max(0,Math.round(centerX-face.bbox.w*.66));
    const right=Math.min(canvas.width-1,Math.round(centerX+face.bbox.w*.66));
    const startY=Math.max(1,Math.round(faceTop));
    const stepX=Math.max(1,Math.round(face.bbox.w/70));

    function rowForegroundFraction(y){
      let fg=0,n=0;
      for(let x=left;x<=right;x+=stepX){
        const i=(y*canvas.width+x)*4;
        if(data[i+3]<32) continue;
        const dr=data[i]-bg[0], dg=data[i+1]-bg[1], db=data[i+2]-bg[2];
        const dist=Math.sqrt(dr*dr+dg*dg+db*db);
        if(dist>34) fg++;
        n++;
      }
      return n ? fg/n : 0;
    }

    let lastSubject=startY;
    let backgroundRun=0;
    for(let y=startY;y>=0;y--){
      const frac=rowForegroundFraction(y);
      if(frac>.10){
        lastSubject=y;
        backgroundRun=0;
      }else if(frac<.035){
        backgroundRun++;
        if(backgroundRun>=5) break;
      }else{
        backgroundRun=Math.max(0,backgroundRun-1);
      }
    }

    // 若背景分析沒有真正往上找到頭髮，使用 Landmark 臉高做保守補償。
    if(lastSubject>=startY-3){
      lastSubject=Math.max(0,startY-face.bbox.h*.28);
    }
    return Math.max(0,lastSubject);
  }

  function mpClampRect(rect,maxW,maxH){
    let {x,y,w,h}=rect;
    if(w>maxW || h>maxH){
      const s=Math.min(maxW/w,maxH/h);
      w*=s; h*=s;
    }
    x=Math.max(0,Math.min(maxW-w,x));
    y=Math.max(0,Math.min(maxH-h,y));
    return {x,y,w,h};
  }

  async function suggestTaiwanHeadshotCropRectMP(inputCanvas){
    const face=await detectMediaPipeFace(inputCanvas);
    if(!face) return null;

    const chin=face.points[152]?.y ?? (face.bbox.y+face.bbox.h);
    const headTop=mpFindHeadTop(inputCanvas,face);
    const headHeight=Math.max(1,chin-headTop);

    const ratio=3.5/4.5;
    const targetHeadRatio=.75;
    const topMarginRatio=.10;
    let cropH=headHeight/targetHeadRatio;
    let cropW=cropH*ratio;
    const faceCenterX=(face.bbox.x+face.bbox.w/2);

    let rect=mpClampRect({
      x:faceCenterX-cropW/2,
      y:headTop-cropH*topMarginRatio,
      w:cropW,h:cropH
    },inputCanvas.width,inputCanvas.height);

    const coverage=headHeight/rect.h;
    const faceAreaRatio=face.area/(inputCanvas.width*inputCanvas.height);
    const reliable=
      face.bbox.w>24 &&
      face.bbox.h>28 &&
      faceAreaRatio>.003 &&
      coverage>=.62 && coverage<=.86;

    return {
      rect,
      face,
      faceCoverage:coverage,
      headTop,
      chin,
      reliable
    };
  }

  function mpBuildSafeSkinPredicate(face,workScale){
    const pts=face.points.map(p=>({x:p.x*workScale,y:p.y*workScale}));
    const oval=mpShrinkPolygon(mpPolygon(pts,MP_FACE_OVAL),.89);
    const faceW=face.bbox.w*workScale;
    const faceH=face.bbox.h*workScale;

    const exclusions=[
      mpExpandedBounds(pts,MP_LEFT_EYE,.42,.80),
      mpExpandedBounds(pts,MP_RIGHT_EYE,.42,.80),
      mpExpandedBounds(pts,MP_LEFT_BROW,.30,.55),
      mpExpandedBounds(pts,MP_RIGHT_BROW,.30,.55),
      mpExpandedBounds(pts,MP_LIPS,.24,.48),
      mpExpandedBounds(pts,MP_NOSE,.28,.24)
    ];

    // 額外排除髮際線最上方與下顎最下緣。
    const ovalBounds=mpBounds(pts,MP_FACE_OVAL);
    const topCut=ovalBounds ? ovalBounds.y+faceH*.08 : -Infinity;
    const bottomCut=ovalBounds ? ovalBounds.y+ovalBounds.h-faceH*.07 : Infinity;

    return (x,y)=>{
      if(!mpPointInPolygon(x,y,oval)) return false;
      if(y<topCut || y>bottomCut) return false;
      for(const r of exclusions){
        if(mpPointInRect(x,y,r)) return false;
      }
      return true;
    };
  }

  async function analyzeFaceBlemishesMediaPipe(inputCanvas,level=1){
    const face=await detectMediaPipeFace(inputCanvas);
    if(!face) return {spots:[],face:null,landmarks:null};

    const maxSide=720;
    const scale=Math.min(1,maxSide/inputCanvas.width,maxSide/inputCanvas.height);
    const w=Math.max(1,Math.round(inputCanvas.width*scale));
    const h=Math.max(1,Math.round(inputCanvas.height*scale));

    const work=document.createElement("canvas");
    work.width=w; work.height=h;
    const ctx=work.getContext("2d",{willReadFrequently:true});
    ctx.drawImage(inputCanvas,0,0,w,h);

    const img=ctx.getImageData(0,0,w,h);
    const data=img.data;
    const count=w*h;
    const lum=new Float32Array(count);
    const reds=new Float32Array(count);
    const greens=new Float32Array(count);
    const blues=new Float32Array(count);
    const skin=new Uint8Array(count);

    for(let p=0;p<count;p++){
      const i=p*4;
      const r=data[i],g=data[i+1],b=data[i+2];
      reds[p]=r; greens[p]=g; blues[p]=b;
      lum[p]=r*.2126+g*.7152+b*.0722;
      skin[p]=isSkinLike(r,g,b)?1:0;
    }

    const intR=buildIntegralFloat(reds,w,h);
    const intG=buildIntegralFloat(greens,w,h);
    const intB=buildIntegralFloat(blues,w,h);
    const intL=buildIntegralFloat(lum,w,h);
    const intSkin=buildIntegralFloat(skin,w,h);

    const safeSkin=mpBuildSafeSkinPredicate(face,scale);
    const sensitivity=Math.max(1,Math.min(3,level|0));
    const colorThreshold=[0,39,33,28][sensitivity];
    const lumThreshold=[0,25,21,18][sensitivity];
    const redThreshold=[0,21,17,14][sensitivity];
    const maxArea=[0,28,44,65][sensitivity];
    const maxDim=[0,12,16,20][sensitivity];
    const radius=Math.max(4,Math.round(Math.min(w,h)*.007));
    const mask=new Uint8Array(count);
    const scoreMap=new Uint8Array(count);

    for(let y=radius+1;y<h-radius-1;y++){
      for(let x=radius+1;x<w-radius-1;x++){
        if(!safeSkin(x,y)) continue;

        const pos=y*w+x;
        const i=pos*4;
        if(data[i+3]<32 || !skin[pos]) continue;

        const x1=x-radius,y1=y-radius,x2=x+radius+1,y2=y+radius+1;
        const area=(x2-x1)*(y2-y1);
        const meanR=integralRectSum(intR,w,x1,y1,x2,y2)/area;
        const meanG=integralRectSum(intG,w,x1,y1,x2,y2)/area;
        const meanB=integralRectSum(intB,w,x1,y1,x2,y2)/area;
        const meanL=integralRectSum(intL,w,x1,y1,x2,y2)/area;
        const skinRatio=integralRectSum(intSkin,w,x1,y1,x2,y2)/area;
        if(skinRatio<.78) continue;

        const r=data[i],g=data[i+1],b=data[i+2];
        const dr=r-meanR,dg=g-meanG,db=b-meanB;
        const colorDiff=Math.sqrt(dr*dr+dg*dg+db*db);
        const lumDiff=Math.abs(lum[pos]-meanL);
        const redness=(r-g)-(meanR-meanG);

        if(
          colorDiff<colorThreshold &&
          lumDiff<lumThreshold &&
          redness<redThreshold
        ) continue;

        // 極深色點保守排除，以避免殘餘睫毛/髮絲。
        if(lum[pos]<48 && lumDiff>30) continue;

        const score=colorDiff*.72+lumDiff*.50+Math.max(0,redness)*.45;
        mask[pos]=1;
        scoreMap[pos]=Math.min(255,Math.round(score));
      }
    }

    const visited=new Uint8Array(count);
    const comps=[];
    const dirs=[[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]];

    for(let y=1;y<h-1;y++){
      for(let x=1;x<w-1;x++){
        const start=y*w+x;
        if(!mask[start] || visited[start]) continue;
        const stack=[start];
        visited[start]=1;
        let area=0,sumX=0,sumY=0,sumScore=0;
        let minX=x,maxX=x,minY=y,maxY=y;

        while(stack.length){
          const pos=stack.pop();
          const py=Math.floor(pos/w);
          const px=pos-py*w;
          area++; sumX+=px; sumY+=py; sumScore+=scoreMap[pos];
          minX=Math.min(minX,px); maxX=Math.max(maxX,px);
          minY=Math.min(minY,py); maxY=Math.max(maxY,py);

          for(const [dx,dy] of dirs){
            const nx=px+dx,ny=py+dy;
            if(nx<=0||nx>=w-1||ny<=0||ny>=h-1) continue;
            const np=ny*w+nx;
            if(mask[np]&&!visited[np]){
              visited[np]=1;
              stack.push(np);
            }
          }
        }

        const bw=maxX-minX+1,bh=maxY-minY+1;
        const longSide=Math.max(bw,bh);
        const shortSide=Math.max(1,Math.min(bw,bh));
        if(area<2 || area>maxArea || longSide>maxDim || longSide/shortSide>2.6) continue;

        const cx=sumX/area,cy=sumY/area;
        if(!safeSkin(cx,cy)) continue;
        const avg=sumScore/area;
        if(avg<colorThreshold+3) continue;

        comps.push({
          x:cx,y:cy,
          radius:Math.max(2.5,longSide*.74),
          score:avg+Math.min(10,area*.2),
          area
        });
      }
    }

    comps.sort((a,b)=>b.score-a.score);
    const maxSpots=[0,22,38,55][sensitivity];
    const selected=[];
    for(const c of comps){
      if(selected.some(s=>Math.hypot(c.x-s.x,c.y-s.y)<Math.max(c.radius,s.radius)*1.25)) continue;
      selected.push(c);
      if(selected.length>=maxSpots) break;
    }

    const inv=1/scale;
    return {
      face:mpFaceOverlayRegion(face),
      landmarks:face.points,
      spots:selected.map(s=>({
        x:s.x*inv,
        y:s.y*inv,
        radius:Math.max(3,s.radius*inv),
        score:s.score,
        area:s.area,
        mode:"face"
      }))
    };
  }

  async function processBlobAutoHeadshotCropMP(blob){
    const img=await blobToImage(blob);
    const c=document.createElement("canvas");
    c.width=img.naturalWidth;
    c.height=img.naturalHeight;
    const ctx=c.getContext("2d",{willReadFrequently:true});
    ctx.drawImage(img,0,0);

    const suggestion=await suggestTaiwanHeadshotCropRectMP(c);
    if(!suggestion || !suggestion.reliable){
      return {success:false,reason:"face-not-detected"};
    }

    const r=suggestion.rect;
    const out=document.createElement("canvas");
    out.width=Math.max(1,Math.round(r.w));
    out.height=Math.max(1,Math.round(r.h));
    out.getContext("2d",{willReadFrequently:true}).drawImage(
      c,r.x,r.y,r.w,r.h,0,0,out.width,out.height
    );
    const outBlob=await canvasToBlob(out,"image/jpeg",.97);
    return outBlob
      ? {success:true,blob:outBlob,faceCoverage:suggestion.faceCoverage}
      : {success:false,reason:"encode-failed"};
  }

  async function runBatchAutoHeadshotCropMP(){
    if(!batchItems.length || batchBusy) return;
    await saveCurrentBatchItem();

    try{
      await initMediaPipeFaceLandmarker();
    }catch{
      showBatchProgress("MediaPipe 人臉模型無法載入，因此未執行批次自動裁切。",true);
      return;
    }

    batchBusy=true;
    updateBatchButtons();
    let success=0,skipped=0;

    try{
      for(let i=0;i<batchItems.length;i++){
        const item=batchItems[i];

        if(item.autoCropped){
          skipped++;
          showBatchProgress(`略過已自動裁切：${i+1} / ${batchItems.length}　${item.file.name}`);
          continue;
        }

        showBatchProgress(`MediaPipe 自動裁成 2 吋：${i+1} / ${batchItems.length}　${item.file.name}`);
        const result=await processBlobAutoHeadshotCropMP(item.editedBlob || item.file);

        if(result.success){
          item.editedBlob=result.blob;
          item.hasTransparency=false;
          item.adjusted=true;
          item.done=false;
          item.autoCropped=true;
          item.cropInfo=`MediaPipe 2 吋裁切，頭部約佔 ${Math.round(result.faceCoverage*100)}%`;
          success++;
        }else{
          item.cropInfo="自動裁切跳過：MediaPipe 未辨識到可靠臉部";
          skipped++;
        }
        renderBatchList();

        // 讓瀏覽器有機會更新 UI。
        await new Promise(r=>setTimeout(r,0));
      }

      showBatchProgress(
        `MediaPipe 批次自動裁切完成：成功 ${success} 張，跳過 ${skipped} 張。`,
        true
      );

      if(batchIndex>=0){
        const current=batchItems[batchIndex];
        await loadBlobIntoEditor(
          current.editedBlob || current.file,
          current.file.name,
          current.filters || defaultBatchFilters(),
          current.autoInfo,
          current.file,
          !!current.hasTransparency
        );
        $("cropRatio").value="0.7777777778";
        $("modeText").textContent=`批次處理：第 ${batchIndex+1} / ${batchItems.length} 張`;
        renderBatchList();
      }
    }catch(err){
      showBatchProgress("MediaPipe 批次自動裁切發生錯誤："+err.message,true);
    }finally{
      batchBusy=false;
      updateBatchButtons();
    }
  }

  async function processBlobSmartCleanMP(blob,level=1){
    const img=await blobToImage(blob);
    const c=document.createElement("canvas");
    c.width=img.naturalWidth;
    c.height=img.naturalHeight;
    const ctx=c.getContext("2d",{willReadFrequently:true});
    ctx.drawImage(img,0,0);

    const analysis=await analyzeFaceBlemishesMediaPipe(c,level);
    const spots=analysis.spots || [];
    let applied=0;
    for(const spot of spots){
      if(healCanvasAt(c,ctx,spot.x,spot.y,spot.radius,.53)) applied++;
    }

    const out=await canvasToBlob(c,"image/jpeg",.97);
    return {
      blob:out,
      detected:spots.length,
      applied,
      faceFound:!!analysis.face
    };
  }

  async function runBatchSmartCleanMP(){
    if(!batchItems.length || batchBusy) return;
    await saveCurrentBatchItem();

    try{
      await initMediaPipeFaceLandmarker();
    }catch{
      showBatchProgress("MediaPipe 人臉模型無法載入，因此未執行批次臉部去污。",true);
      return;
    }

    batchBusy=true;
    updateBatchButtons();
    let totalDetected=0,totalApplied=0,faceMissing=0;

    try{
      for(let i=0;i<batchItems.length;i++){
        const item=batchItems[i];
        showBatchProgress(`MediaPipe 臉部去污：${i+1} / ${batchItems.length}　${item.file.name}`);

        const result=await processBlobSmartCleanMP(item.editedBlob || item.file,1);
        if(result.faceFound && result.blob){
          item.editedBlob=result.blob;
          item.hasTransparency=false;
          item.adjusted=true;
          item.done=false;
          totalDetected+=result.detected;
          totalApplied+=result.applied;
        }else{
          faceMissing++;
        }
        renderBatchList();
        await new Promise(r=>setTimeout(r,0));
      }

      showBatchProgress(
        `批次臉部去污完成：修補 ${totalApplied} 個候選斑點；另有 ${faceMissing} 張未辨識到臉部而跳過。`,
        true
      );

      if(batchIndex>=0){
        const current=batchItems[batchIndex];
        await loadBlobIntoEditor(
          current.editedBlob || current.file,
          current.file.name,
          current.filters || defaultBatchFilters(),
          current.autoInfo,
          current.file,
          !!current.hasTransparency
        );
        $("modeText").textContent=`批次處理：第 ${batchIndex+1} / ${batchItems.length} 張`;
        renderBatchList();
      }
    }catch(err){
      showBatchProgress("MediaPipe 批次臉部去污發生錯誤："+err.message,true);
    }finally{
      batchBusy=false;
      updateBatchButtons();
    }
  }
  const $ = id => document.getElementById(id);

  const fileInput = $('fileInput');
  const batchFileInput = $('batchFileInput');
  const preview = $('preview');
  const pctx = preview.getContext('2d', { willReadFrequently: true });
  const overlay = $('overlay');
  const octx = overlay.getContext('2d');
  const canvasWrap = $('canvasWrap');
  const empty = $('empty');
  const app = $('app');
  const dropZone = $('dropZone');
  const stageArea = $('stageArea');
  const mainLayout = $('mainLayout');

  const source = document.createElement('canvas');
  const sctx = source.getContext('2d', { willReadFrequently: true });

  let originalImage = null;
  let originalName = 'photo';
  let history = [];
  let historyIndex = -1;
  let cropMode = false;
  let cropRect = null;
  let dragStart = null;
  let cropDragMode = null;
  let cropStartRect = null;
  let cropHover = null;
  let healMode = false;
  let healCursor = null;
  let renderToken = 0;
  let zoomLevel = 100;
  let zoomMode = 'fit';
  let editorMode = 'single';
  let batchItems = [];
  let batchIndex = -1;
  let batchBusy = false;
  let loadingBatchItem = false;
  let sourceDirty = false;
  let smartSpots = [];
  let smartFaceRegion = null;
  let smartAnalysisMode = 'face';
  let compareOriginalImage = null;
  let compareHolding = false;
  let compareSavedModeText = '';
  let sourceHasTransparency = false;

  const controls = [
    'rotateLeftBtn','rotateRightBtn','autoHeadshotCropBtn','cropBtn','cropRatio','brightness','contrast','saturation','sharpen',
    'autoBtn','quickBrightBtn','compareHoldBtn','resetFilterBtn','smartAnalyzeBtn','smartCleanMode','smartCleanLevel','smartClearBtn',
    'healBtn','brush','bgOutputMode','bgFeather','removeBgBtn','downloadJpgBtn','downloadPngBtn','resetAllBtn',
    'zoomOutBtn','zoomInBtn','zoomRange','fitZoomBtn'
  ];

  function canvasToBlob(canvas, type='image/jpeg', quality=.97){
    return new Promise(resolve => canvas.toBlob(resolve, type, quality));
  }

  function blobToImage(blob){
    return new Promise((resolve,reject)=>{
      const url=URL.createObjectURL(blob);
      const img=new Image();
      img.onload=()=>{
        URL.revokeObjectURL(url);
        resolve(img);
      };
      img.onerror=()=>{
        URL.revokeObjectURL(url);
        reject(new Error('圖片載入失敗'));
      };
      img.src=url;
    });
  }

  function getCurrentFilterState(){
    return {
      brightness:+$('brightness').value,
      contrast:+$('contrast').value,
      saturation:+$('saturation').value,
      sharpen:+$('sharpen').value
    };
  }

  function setFilterState(state){
    const s=state || {brightness:100,contrast:100,saturation:100,sharpen:0};
    $('brightness').value=s.brightness ?? 100;
    $('contrast').value=s.contrast ?? 100;
    $('saturation').value=s.saturation ?? 100;
    $('sharpen').value=s.sharpen ?? 0;
    syncLabels();
  }

  function defaultBatchFilters(){
    return {brightness:100,contrast:100,saturation:100,sharpen:0};
  }

  function getBatchStatusText(item){
    if(item.done) return '已完成';
    if(item.adjusted || item.editedBlob) return '已調整';
    return '待處理';
  }

  function showBatchProgress(text, show=true){
    const el=$('batchProgress');
    el.textContent=text || '';
    el.classList.toggle('show', !!show);
  }

  function updateBatchButtons(){
    const has=batchItems.length>0;
    const active=batchIndex>=0 && batchIndex<batchItems.length;
    $('batchPrevBtn').disabled=!active || batchIndex<=0 || batchBusy;
    $('batchNextBtn').disabled=!active || batchIndex>=batchItems.length-1 || batchBusy;
    $('batchCropBtn').disabled=!has || batchBusy;
    $('batchAutoBtn').disabled=!has || batchBusy;
    $('batchCleanBtn').disabled=!has || batchBusy;
    $('batchBgWhiteBtn').disabled=!has || batchBusy;
    $('batchBgTransparentBtn').disabled=!has || batchBusy;
    $('batchDoneBtn').disabled=!active || batchBusy;
    $('batchZipBtn').disabled=!has || batchBusy;
    $('batchClearBtn').disabled=!has || batchBusy;
    $('batchDoneBtn').textContent=active && batchItems[batchIndex].done
      ? '↩ 取消完成'
      : '✓ 標記完成';
  }

  function renderBatchList(){
    const list=$('batchList');
    list.innerHTML='';
    batchItems.forEach((item,index)=>{
      const row=document.createElement('div');
      row.className='batch-item' + (index===batchIndex ? ' active' : '');
      row.dataset.index=index;

      const img=document.createElement('img');
      img.className='batch-thumb';
      img.src=item.thumbUrl;
      img.alt='';

      const main=document.createElement('div');
      main.className='batch-item-main';

      const name=document.createElement('div');
      name.className='batch-name';
      name.title=item.file.name;
      name.textContent=item.file.name;

      const status=document.createElement('span');
      status.className='batch-status' + (item.done ? ' done' : (item.adjusted || item.editedBlob ? ' adjusted' : ''));
      status.textContent=getBatchStatusText(item);

      main.append(name,status);
      row.append(img,main);
      row.addEventListener('click',()=>loadBatchItem(index));
      list.appendChild(row);
    });

    $('batchCount').textContent=`共 ${batchItems.length} 張`;
    $('batchDoneCount').textContent=`完成 ${batchItems.filter(x=>x.done).length} 張`;
    updateBatchButtons();
  }

  function touchCurrentBatchItem(){
    if(editorMode!=='batch' || loadingBatchItem) return;
    const item=batchItems[batchIndex];
    if(!item) return;
    if(!item.done) item.adjusted=true;
    item.filters=getCurrentFilterState();
    item.autoInfo=$('autoInfo').textContent;
    renderBatchList();
  }

  async function saveCurrentBatchItem(){
    if(editorMode!=='batch' || loadingBatchItem) return;
    const item=batchItems[batchIndex];
    if(!item || !source.width) return;

    item.filters=getCurrentFilterState();
    item.autoInfo=$('autoInfo').textContent;

    if(sourceDirty){
      const type=sourceHasTransparency ? 'image/png' : 'image/jpeg';
      const blob=await canvasToBlob(source,type,.97);
      if(blob){
        item.editedBlob=blob;
        item.hasTransparency=sourceHasTransparency;
      }
      sourceDirty=false;
    }
  }

  async function setCompareOriginalFromBlob(blob){
    if(!blob){
      compareOriginalImage=null;
      return;
    }
    try{
      compareOriginalImage=await blobToImage(blob);
    }catch{
      compareOriginalImage=null;
    }
  }

  function drawImageContain(ctx,img,w,h){
    ctx.clearRect(0,0,w,h);
    ctx.fillStyle='#ffffff';
    ctx.fillRect(0,0,w,h);
    if(!img || !img.naturalWidth || !img.naturalHeight) return;

    const scale=Math.min(w/img.naturalWidth,h/img.naturalHeight);
    const dw=Math.max(1,Math.round(img.naturalWidth*scale));
    const dh=Math.max(1,Math.round(img.naturalHeight*scale));
    const dx=Math.round((w-dw)/2);
    const dy=Math.round((h-dh)/2);
    ctx.drawImage(img,dx,dy,dw,dh);
  }

  function renderBeforeComparison(){
    if(!compareOriginalImage || !preview.width || !preview.height) return;
    drawImageContain(pctx,compareOriginalImage,preview.width,preview.height);
    octx.clearRect(0,0,overlay.width,overlay.height);
    $('modeText').textContent='修改前（放開按鈕返回修改後）';
  }

  function startCompareBefore(){
    if(!compareOriginalImage || !source.width) return;
    compareHolding=true;
    compareSavedModeText=$('modeText').textContent;
    $('compareHoldBtn').classList.add('active');
    renderBeforeComparison();
  }

  function endCompareBefore(){
    if(!compareHolding) return;
    compareHolding=false;
    $('compareHoldBtn').classList.remove('active');
    renderPreview().then(()=>{
      if(compareSavedModeText) $('modeText').textContent=compareSavedModeText;
    });
  }

  async function loadBlobIntoEditor(blob, name, filterState, autoInfoText, compareBlob=null, hasTransparency=false){
    loadingBatchItem=true;
    try{
      const img=await blobToImage(blob);
      originalImage=img;
      originalName=name.replace(/\.[^.]+$/,'') || 'photo';
      compareHolding=false;
      sourceHasTransparency=!!hasTransparency;
      compareOriginalImage = compareBlob ? await blobToImage(compareBlob) : img;

      source.width=img.naturalWidth;
      source.height=img.naturalHeight;
      sctx.clearRect(0,0,source.width,source.height);
      sctx.drawImage(img,0,0);

      history=[];
      historyIndex=-1;
      zoomMode='fit';
      zoomLevel=100;
      healCursor=null;
      smartSpots=[];
      smartFaceRegion=null;
      cropRect=null;
      cropMode=false;
      dragStart=null;
      cropDragMode=null;
      cropStartRect=null;
      cropHover=null;
      preview.style.cursor='default';
      $('cropBtn').classList.remove('active');
      $('healBtn').classList.remove('active');
      $('healBtn').textContent='去污筆：關閉';
      healMode=false;

      setFilterState(filterState || defaultBatchFilters());
      $('autoInfo').textContent=autoInfoText || '智慧自動提亮會先分析照片的明暗分布，再決定提亮幅度；快速提亮則使用固定參數。';
      if($('smartCleanInfo')){
        $('smartCleanInfo').textContent=$('smartCleanMode').value==='face'
          ? '「臉部斑點」會先估算主要臉部範圍，再排除眼睛、眉毛、鼻孔、嘴巴、耳朵、脖子與衣服，只分析安全皮膚區域。'
          : '「照片灰塵／小型損傷」會分析整張照片中的小黑點、小白點與小型掃描髒污。';
      }
      $('smartApplyBtn').disabled=true;

      pushHistory();
      sourceDirty=false;
      canvasWrap.hidden=false;
      empty.hidden=true;
      setEnabled(true);
      updateMeta();
      await renderPreview();
    }finally{
      loadingBatchItem=false;
    }
  }

  async function loadBatchItem(index){
    if(batchBusy || index<0 || index>=batchItems.length) return;
    if(editorMode!=='batch') await switchEditorMode('batch');
    if(batchIndex===index && source.width) return;

    await saveCurrentBatchItem();
    batchIndex=index;
    const item=batchItems[index];
    const blob=item.editedBlob || item.file;

    await loadBlobIntoEditor(
      blob,
      item.file.name,
      item.filters || defaultBatchFilters(),
      item.autoInfo,
      item.file,
      !!item.hasTransparency
    );

    $('modeText').textContent=`批次處理：第 ${index+1} / ${batchItems.length} 張`;
    renderBatchList();
  }

  function clearEditor(){
    source.width=0;
    source.height=0;
    preview.width=1;
    preview.height=1;
    overlay.width=1;
    overlay.height=1;
    canvasWrap.hidden=true;
    empty.hidden=false;
    empty.innerHTML='<div style="font-size:44px">🖼️</div>' +
      (editorMode==='batch'
        ? '請先加入多張照片'
        : '點「上傳照片」或將照片拖曳到這裡');
    $('meta').textContent='尚未載入照片';
    $('modeText').textContent=editorMode==='batch' ? '批次處理' : '預覽';
    setEnabled(false);
    sourceDirty=false;
    sourceHasTransparency=false;
    compareOriginalImage=null;
    compareHolding=false;
  }

  async function switchEditorMode(mode){
    if(mode===editorMode) return;

    if(editorMode==='batch'){
      await saveCurrentBatchItem();
    }

    editorMode=mode;
    const batch=mode==='batch';

    $('singleModeBtn').classList.toggle('active',!batch);
    $('batchModeBtn').classList.toggle('active',batch);
    $('batchPanel').hidden=!batch;
    if($('singleInfoPanel')) $('singleInfoPanel').hidden=batch;
    mainLayout.classList.toggle('batch-mode',batch);
    app.classList.toggle('batch-active',batch);
    $('openBtn').textContent=batch ? '加入照片' : '上傳照片';
    if(typeof activateRibbonTab==='function'){
      activateRibbonTab(batch ? 'batch' : 'home');
    }
    $('stageHint').textContent=batch
      ? '批次模式：上一張／下一張快速檢查；局部修圖會在切換時保存'
      : '可拖曳圖片到中央區域；放大僅影響檢視，不會改變照片尺寸';

    if(batch){
      if(batchItems.length){
        batchIndex=-1;
        await loadBatchItem(0);
      }else{
        batchIndex=-1;
        clearEditor();
        renderBatchList();
      }
    }else{
      batchIndex=-1;
      clearEditor();
    }
  }

  async function addBatchFiles(fileList){
    const files=[...fileList].filter(f=>f && f.type && f.type.startsWith('image/'));
    if(!files.length) return;

    for(const file of files){
      batchItems.push({
        file,
        thumbUrl:URL.createObjectURL(file),
        editedBlob:null,
        filters:defaultBatchFilters(),
        autoInfo:'',
        adjusted:false,
        done:false,
        autoCropped:false,
        hasTransparency:false
      });
    }

    renderBatchList();
    if(batchIndex<0){
      await loadBatchItem(0);
    }
  }

  async function analyzeBlobBrightness(blob){
    const img=await blobToImage(blob);
    const maxSide=180;
    const scale=Math.min(1,maxSide/img.naturalWidth,maxSide/img.naturalHeight);
    const w=Math.max(1,Math.round(img.naturalWidth*scale));
    const h=Math.max(1,Math.round(img.naturalHeight*scale));

    const c=document.createElement('canvas');
    c.width=w;
    c.height=h;
    const ctx=c.getContext('2d',{willReadFrequently:true});
    ctx.drawImage(img,0,0,w,h);

    const data=ctx.getImageData(0,0,w,h).data;
    const overall=new Uint32Array(256);
    const center=new Uint32Array(256);
    let overallCount=0, centerCount=0;
    const x1=w*.20, x2=w*.80, y1=h*.10, y2=h*.90;

    for(let y=0;y<h;y++){
      for(let x=0;x<w;x++){
        const i=(y*w+x)*4;
        if(data[i+3]<16) continue;
        const lum=Math.max(0,Math.min(255,Math.round(
          data[i]*0.2126 + data[i+1]*0.7152 + data[i+2]*0.0722
        )));
        overall[lum]++;
        overallCount++;
        if(x>=x1 && x<=x2 && y>=y1 && y<=y2){
          center[lum]++;
          centerCount++;
        }
      }
    }

    const allStats=histogramStats(overall,overallCount);
    const centerStats=histogramStats(center,centerCount || overallCount);
    const reference=centerCount
      ? centerStats.mean*.72 + allStats.mean*.28
      : allStats.mean;

    return {
      reference,
      dynamicRange:Math.max(1,allStats.p90-allStats.p10),
      overall:allStats,
      center:centerStats
    };
  }

  async function processBlobAutoHeadshotCrop(blob){
    const img=await blobToImage(blob);
    const c=document.createElement('canvas');
    c.width=img.naturalWidth;
    c.height=img.naturalHeight;
    const ctx=c.getContext('2d',{willReadFrequently:true});
    ctx.drawImage(img,0,0);

    const suggestion=suggestTaiwanHeadshotCropRectForCanvas(c);
    if(!suggestion || !suggestion.reliable){
      return {success:false,reason:'face-not-reliable'};
    }

    const r=suggestion.rect;
    if(r.w<20 || r.h<20){
      return {success:false,reason:'crop-too-small'};
    }

    const out=document.createElement('canvas');
    out.width=Math.max(1,Math.round(r.w));
    out.height=Math.max(1,Math.round(r.h));
    const o=out.getContext('2d',{willReadFrequently:true});
    o.drawImage(
      c,
      r.x,r.y,r.w,r.h,
      0,0,out.width,out.height
    );

    const outBlob=await canvasToBlob(out,'image/jpeg',.97);
    if(!outBlob) return {success:false,reason:'encode-failed'};

    return {
      success:true,
      blob:outBlob,
      faceCoverage:suggestion.faceCoverage
    };
  }

  async function runBatchAutoHeadshotCrop(){
    if(!batchItems.length || batchBusy) return;
    await saveCurrentBatchItem();

    batchBusy=true;
    updateBatchButtons();

    let success=0;
    let skipped=0;

    try{
      for(let i=0;i<batchItems.length;i++){
        const item=batchItems[i];

        if(item.autoCropped){
          skipped++;
          showBatchProgress(`略過已自動裁切：${i+1} / ${batchItems.length}　${item.file.name}`);
          continue;
        }

        showBatchProgress(`自動裁成 2 吋：${i+1} / ${batchItems.length}　${item.file.name}`);

        const result=await processBlobAutoHeadshotCrop(item.editedBlob || item.file);
        if(result.success){
          item.editedBlob=result.blob;
          item.adjusted=true;
          item.done=false;
          item.autoCropped=true;
          item.cropInfo=`2 吋自動裁切，臉部約佔 ${Math.round(result.faceCoverage*100)}%`;
          success++;
        }else{
          item.cropInfo='自動裁切跳過：未可靠找到臉部';
          skipped++;
        }

        renderBatchList();
      }

      showBatchProgress(
        `全批自動裁切完成：成功 ${success} 張，跳過 ${skipped} 張。跳過的照片請逐張手動裁切。`,
        true
      );

      if(batchIndex>=0){
        const current=batchItems[batchIndex];
        await loadBlobIntoEditor(
          current.editedBlob || current.file,
          current.file.name,
          current.filters || defaultBatchFilters(),
          current.autoInfo,
          current.file
        );
        $('cropRatio').value='0.7777777778';
        $('modeText').textContent=`批次處理：第 ${batchIndex+1} / ${batchItems.length} 張`;
        renderBatchList();
      }
    }catch(err){
      showBatchProgress('批次自動裁切發生錯誤：'+err.message,true);
    }finally{
      batchBusy=false;
      updateBatchButtons();
    }
  }

  async function runBatchSmartBright(){
    if(!batchItems.length || batchBusy) return;
    await saveCurrentBatchItem();
    batchBusy=true;
    updateBatchButtons();

    try{
      for(let i=0;i<batchItems.length;i++){
        const item=batchItems[i];
        showBatchProgress(`智慧分析中：${i+1} / ${batchItems.length}　${item.file.name}`);
        const stats=await analyzeBlobBrightness(item.editedBlob || item.file);
        const adj=calculateSmartAdjustments(stats);
        item.filters={
          brightness:adj.brightness,
          contrast:adj.contrast,
          saturation:adj.saturation,
          sharpen:adj.sharpen
        };
        item.adjusted=true;
        const refPct=Math.round(stats.reference/255*100);
        item.autoInfo=
          `批次分析：${adj.label}（基準亮度約 ${refPct}%）。`+
          `亮度 ${adj.brightness}%、對比 ${adj.contrast}%、`+
          `飽和度 ${adj.saturation}%、銳化 ${adj.sharpen}。`;
        renderBatchList();
      }

      showBatchProgress(`已完成 ${batchItems.length} 張智慧提亮。`,true);
      if(batchIndex>=0){
        const current=batchItems[batchIndex];
        setFilterState(current.filters);
        $('autoInfo').textContent=current.autoInfo;
        await renderPreview();
      }
    }catch(err){
      showBatchProgress('批次智慧提亮發生錯誤：' + err.message,true);
    }finally{
      batchBusy=false;
      updateBatchButtons();
    }
  }


  async function runBatchSmartClean(){
    if(!batchItems.length || batchBusy) return;
    await saveCurrentBatchItem();
    batchBusy=true;
    updateBatchButtons();

    let totalDetected=0;
    let totalApplied=0;

    try{
      for(let i=0;i<batchItems.length;i++){
        const item=batchItems[i];
        showBatchProgress(`臉部智慧去污中：${i+1} / ${batchItems.length}　${item.file.name}`);
        const result=await processBlobSmartClean(item.editedBlob || item.file,1,'face');
        if(result.blob){
          item.editedBlob=result.blob;
          item.adjusted=true;
          item.done=false;
          totalDetected+=result.detected;
          totalApplied+=result.applied;
        }
        renderBatchList();
      }

      showBatchProgress(
        `批次臉部去污完成：共偵測 ${totalDetected} 個臉部候選斑點，已保守修補 ${totalApplied} 個。未找到明確臉部的照片不會亂修，仍建議逐張快速檢查。`,
        true
      );

      if(batchIndex>=0){
        const current=batchItems[batchIndex];
        await loadBlobIntoEditor(
          current.editedBlob || current.file,
          current.file.name,
          current.filters || defaultBatchFilters(),
          current.autoInfo,
          current.file
        );
        $('modeText').textContent=`批次處理：第 ${batchIndex+1} / ${batchItems.length} 張`;
        renderBatchList();
      }
    }catch(err){
      showBatchProgress('批次臉部去污發生錯誤：'+err.message,true);
    }finally{
      batchBusy=false;
      updateBatchButtons();
    }
  }

  async function processBlobRemoveBackgroundMP(blob,outputMode="white",featherPx=1.5){
    const img=await blobToImage(blob);
    const c=document.createElement("canvas");
    c.width=img.naturalWidth;
    c.height=img.naturalHeight;
    const ctx=c.getContext("2d",{willReadFrequently:true});
    ctx.clearRect(0,0,c.width,c.height);
    ctx.drawImage(img,0,0);

    const result=await runImageSegmentation(c);
    const mask=buildForegroundMaskCanvas(
      result,
      c.width,
      c.height,
      featherPx
    );

    const out=document.createElement("canvas");
    out.width=c.width;
    out.height=c.height;
    const o=out.getContext("2d");

    o.clearRect(0,0,out.width,out.height);
    o.drawImage(c,0,0);
    o.globalCompositeOperation="destination-in";
    o.drawImage(mask,0,0);
    o.globalCompositeOperation="source-over";

    let type="image/png";
    if(outputMode==="white"){
      o.globalCompositeOperation="destination-over";
      o.fillStyle="#ffffff";
      o.fillRect(0,0,out.width,out.height);
      o.globalCompositeOperation="source-over";
      type="image/jpeg";
    }

    const blobOut=await canvasToBlob(out,type,.97);
    return blobOut
      ? {
          success:true,
          blob:blobOut,
          hasTransparency:outputMode==="transparent"
        }
      : {
          success:false,
          reason:"encode-failed",
          hasTransparency:false
        };
  }

  async function runBatchRemoveBackgroundMP(outputMode="white"){
    if(!batchItems.length || batchBusy) return;
    await saveCurrentBatchItem();

    try{
      await initMediaPipeImageSegmenter();
    }catch(err){
      showBatchProgress(
        "MediaPipe 人物去背模型無法載入，因此未執行批次去背：" +
        (err?.message || "未知錯誤"),
        true
      );
      return;
    }

    batchBusy=true;
    updateBatchButtons();

    let success=0;
    let skipped=0;
    const modeText=outputMode==="transparent" ? "透明" : "白色";

    try{
      for(let i=0;i<batchItems.length;i++){
        const item=batchItems[i];

        showBatchProgress(
          `批次去背（${modeText}）：${i+1} / ${batchItems.length}　${item.file.name}`
        );

        try{
          const result=await processBlobRemoveBackgroundMP(
            item.editedBlob || item.file,
            outputMode,
            1.5
          );

          if(result.success && result.blob){
            item.editedBlob=result.blob;
            item.hasTransparency=!!result.hasTransparency;
            item.adjusted=true;
            item.done=false;
            item.bgInfo=`批次去背（${modeText}）`;
            success++;
          }else{
            item.bgInfo=`批次去背（${modeText}）失敗`;
            skipped++;
          }
        }catch(itemError){
          console.warn("單張批次去背失敗",item.file.name,itemError);
          item.bgInfo=`批次去背（${modeText}）跳過`;
          skipped++;
        }

        renderBatchList();
        await new Promise(r=>setTimeout(r,0));
      }

      showBatchProgress(
        `批次去背（${modeText}）完成：成功 ${success} 張，跳過 ${skipped} 張。` +
        (outputMode==="transparent"
          ? " 透明照片下載 ZIP 時會以 PNG 輸出。"
          : ""),
        true
      );

      if(batchIndex>=0){
        const current=batchItems[batchIndex];
        await loadBlobIntoEditor(
          current.editedBlob || current.file,
          current.file.name,
          current.filters || defaultBatchFilters(),
          current.autoInfo,
          current.file,
          !!current.hasTransparency
        );
        $('modeText').textContent=
          `批次處理：第 ${batchIndex+1} / ${batchItems.length} 張`;
        renderBatchList();
      }
    }catch(err){
      showBatchProgress(
        `批次去背（${modeText}）發生錯誤：` +
        (err?.message || "未知錯誤"),
        true
      );
    }finally{
      batchBusy=false;
      updateBatchButtons();
    }
  }

  function snapshotOutputSettings(){
    return {
      preset:$('presetSize').value,
      outW:parseInt($('outW').value)||0,
      outH:parseInt($('outH').value)||0,
      keepRatio:$('keepRatio').checked,
      quality:+$('quality').value/100
    };
  }

  async function renderBatchItemBlob(item, settings){
    const img=await blobToImage(item.editedBlob || item.file);
    const srcC=document.createElement('canvas');
    srcC.width=img.naturalWidth;
    srcC.height=img.naturalHeight;
    const srcCtx=srcC.getContext('2d',{willReadFrequently:true});
    srcCtx.clearRect(0,0,srcC.width,srcC.height);
    srcCtx.drawImage(img,0,0);

    let outW=srcC.width, outH=srcC.height;
    if(settings.preset!=='original'){
      outW=Math.max(1,settings.outW || srcC.width);
      outH=Math.max(1,settings.outH || srcC.height);
    }

    const transparent=!!item.hasTransparency;
    const outputType=transparent ? 'image/png' : 'image/jpeg';

    const base=document.createElement('canvas');
    base.width=outW;
    base.height=outH;
    const b=base.getContext('2d',{willReadFrequently:true});
    const f=item.filters || defaultBatchFilters();

    b.clearRect(0,0,outW,outH);
    b.filter=`brightness(${f.brightness}%) contrast(${f.contrast}%) saturate(${f.saturation}%)`;

    if(settings.keepRatio && settings.preset!=='original'){
      const scale=Math.min(outW/srcC.width,outH/srcC.height);
      const dw=Math.round(srcC.width*scale);
      const dh=Math.round(srcC.height*scale);
      const dx=Math.round((outW-dw)/2);
      const dy=Math.round((outH-dh)/2);

      if(!transparent){
        b.fillStyle='#ffffff';
        b.fillRect(0,0,outW,outH);
      }

      b.drawImage(srcC,dx,dy,dw,dh);
    }else{
      b.drawImage(srcC,0,0,outW,outH);
    }
    b.filter='none';

    if(f.sharpen>0){
      const data=b.getImageData(0,0,outW,outH);
      applySharpen(data,outW,outH,f.sharpen);
      b.putImageData(data,0,0);
    }

    const blob=await canvasToBlob(base,outputType,settings.quality);
    return {
      blob,
      extension:transparent ? 'png' : 'jpg',
      transparent
    };
  }

  const crcTable=(()=>{
    const table=new Uint32Array(256);
    for(let n=0;n<256;n++){
      let c=n;
      for(let k=0;k<8;k++) c=(c&1) ? (0xedb88320^(c>>>1)) : (c>>>1);
      table[n]=c>>>0;
    }
    return table;
  })();

  function crc32(bytes){
    let c=0xffffffff;
    for(let i=0;i<bytes.length;i++){
      c=crcTable[(c^bytes[i])&0xff]^(c>>>8);
    }
    return (c^0xffffffff)>>>0;
  }

  function zipDateTime(date=new Date()){
    const year=Math.max(1980,date.getFullYear());
    const time=(date.getHours()<<11) | (date.getMinutes()<<5) | Math.floor(date.getSeconds()/2);
    const day=((year-1980)<<9) | ((date.getMonth()+1)<<5) | date.getDate();
    return {time,day};
  }

  function writeU16(view,offset,value){ view.setUint16(offset,value,true); }
  function writeU32(view,offset,value){ view.setUint32(offset,value>>>0,true); }

  function makeZip(files){
    const encoder=new TextEncoder();
    const localParts=[];
    const centralParts=[];
    let offset=0;
    const dt=zipDateTime();

    for(const file of files){
      const nameBytes=encoder.encode(file.name);
      const data=file.bytes;
      const crc=crc32(data);

      const local=new Uint8Array(30+nameBytes.length);
      const lv=new DataView(local.buffer);
      writeU32(lv,0,0x04034b50);
      writeU16(lv,4,20);
      writeU16(lv,6,0x0800);
      writeU16(lv,8,0);
      writeU16(lv,10,dt.time);
      writeU16(lv,12,dt.day);
      writeU32(lv,14,crc);
      writeU32(lv,18,data.length);
      writeU32(lv,22,data.length);
      writeU16(lv,26,nameBytes.length);
      writeU16(lv,28,0);
      local.set(nameBytes,30);

      localParts.push(local,data);

      const central=new Uint8Array(46+nameBytes.length);
      const cv=new DataView(central.buffer);
      writeU32(cv,0,0x02014b50);
      writeU16(cv,4,20);
      writeU16(cv,6,20);
      writeU16(cv,8,0x0800);
      writeU16(cv,10,0);
      writeU16(cv,12,dt.time);
      writeU16(cv,14,dt.day);
      writeU32(cv,16,crc);
      writeU32(cv,20,data.length);
      writeU32(cv,24,data.length);
      writeU16(cv,28,nameBytes.length);
      writeU16(cv,30,0);
      writeU16(cv,32,0);
      writeU16(cv,34,0);
      writeU16(cv,36,0);
      writeU32(cv,38,0);
      writeU32(cv,42,offset);
      central.set(nameBytes,46);
      centralParts.push(central);

      offset += local.length + data.length;
    }

    const centralSize=centralParts.reduce((sum,p)=>sum+p.length,0);
    const end=new Uint8Array(22);
    const ev=new DataView(end.buffer);
    writeU32(ev,0,0x06054b50);
    writeU16(ev,4,0);
    writeU16(ev,6,0);
    writeU16(ev,8,files.length);
    writeU16(ev,10,files.length);
    writeU32(ev,12,centralSize);
    writeU32(ev,16,offset);
    writeU16(ev,20,0);

    return new Blob([...localParts,...centralParts,end],{type:'application/zip'});
  }

  function cleanZipBaseName(filename){
    return filename.replace(/\.[^.]+$/,'').replace(/[\\/:*?"<>|]/g,'_').trim() || 'photo';
  }

  async function exportBatchZip(){
    if(!batchItems.length || batchBusy) return;
    await saveCurrentBatchItem();

    batchBusy=true;
    updateBatchButtons();
    const settings=snapshotOutputSettings();
    const used=new Map();
    const zipFiles=[];

    try{
      for(let i=0;i<batchItems.length;i++){
        const item=batchItems[i];
        const output=await renderBatchItemBlob(item,settings);
        if(!output?.blob) throw new Error('批次照片產生失敗');

        showBatchProgress(
          `輸出 ${output.extension.toUpperCase()}：${i+1} / ${batchItems.length}　${item.file.name}`
        );

        let base=cleanZipBaseName(item.file.name) + '_edited';
        const count=(used.get(base)||0)+1;
        used.set(base,count);
        if(count>1) base += '_' + count;

        const bytes=new Uint8Array(await output.blob.arrayBuffer());
        zipFiles.push({
          name:base+'.'+output.extension,
          bytes
        });
      }

      showBatchProgress('正在建立 ZIP 檔案…');
      const zip=makeZip(zipFiles);
      const a=document.createElement('a');
      a.href=URL.createObjectURL(zip);
      const now=new Date();
      const pad=n=>String(n).padStart(2,'0');
      a.download=`會員照片批次_${now.getFullYear()}${pad(now.getMonth()+1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}.zip`;
      document.body.appendChild(a);
      a.click();
      setTimeout(()=>{
        URL.revokeObjectURL(a.href);
        a.remove();
      },1500);

      showBatchProgress(`已完成 ${batchItems.length} 張照片並建立 ZIP；透明背景照片以 PNG 輸出。`,true);
    }catch(err){
      showBatchProgress('批次匯出發生錯誤：' + err.message,true);
    }finally{
      batchBusy=false;
      updateBatchButtons();
    }
  }

  async function clearBatchItems(){
    if(batchBusy) return;
    await saveCurrentBatchItem();
    batchItems.forEach(item=>URL.revokeObjectURL(item.thumbUrl));
    batchItems=[];
    batchIndex=-1;
    renderBatchList();
    clearEditor();
    showBatchProgress('',false);
  }

  function setEnabled(on){
    controls.forEach(id => $(id).disabled = !on);
    $('smartApplyBtn').disabled = !on || smartSpots.length===0;
    updateHistoryButtons();
  }

  function filters(){
    return {
      brightness:+$('brightness').value,
      contrast:+$('contrast').value,
      saturation:+$('saturation').value,
      sharpen:+$('sharpen').value
    };
  }

  function resetFilterValues(){
    $('brightness').value = 100;
    $('contrast').value = 100;
    $('saturation').value = 100;
    $('sharpen').value = 0;
    syncLabels();
  }

  function resetAutoInfo(){
    const info = $('autoInfo');
    if(info){
      info.textContent='智慧自動提亮會先分析照片的明暗分布，再決定提亮幅度；快速提亮則使用固定參數。';
    }
  }

  function resetSmartCleanInfo(){
    smartSpots=[];
    smartFaceRegion=null;
    smartAnalysisMode=$('smartCleanMode') ? $('smartCleanMode').value : 'face';
    if($('smartCleanInfo')){
      $('smartCleanInfo').textContent=smartAnalysisMode==='face'
        ? '「臉部斑點」會先估算主要臉部範圍，再排除眼睛、眉毛、鼻孔、嘴巴、耳朵、脖子與衣服，只分析安全皮膚區域。'
        : '「照片灰塵／小型損傷」會分析整張照片中的小黑點、小白點與小型掃描髒污，不用來判斷青春痘或老人斑。';
    }
    if($('smartApplyBtn')) $('smartApplyBtn').disabled=true;
    drawOverlay();
  }

  function smartCleanLevelLabel(){
    return ({1:'保守',2:'標準',3:'較強'})[+$('smartCleanLevel').value] || '保守';
  }

  function syncLabels(){
    $('brightnessVal').textContent = $('brightness').value + '%';
    $('contrastVal').textContent = $('contrast').value + '%';
    $('saturationVal').textContent = $('saturation').value + '%';
    $('sharpenVal').textContent = $('sharpen').value;
    $('brushVal').textContent = $('brush').value + ' px';
    $('qualityVal').textContent = $('quality').value + '%';
    $('zoomVal').textContent = Math.round(zoomLevel) + '%';
    if($('smartCleanLevelVal')) $('smartCleanLevelVal').textContent=smartCleanLevelLabel();
  }

  function updateMeta(){
    if(!source.width) return;
    const mp = (source.width * source.height / 1_000_000).toFixed(1);
    $('meta').innerHTML = `
      <b>${escapeHtml(originalName)}</b><br>
      目前尺寸：${source.width} × ${source.height}px<br>
      約 ${mp} 百萬像素
    `;
    if($('presetSize').value === 'original'){
      $('outW').value = source.width;
      $('outH').value = source.height;
    }
  }

  function escapeHtml(str){
    return String(str).replace(/[&<>"']/g, m => ({
      '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
    }[m]));
  }

  function calcPreviewSize(){
    const maxW = 1200;
    const maxH = 900;
    const scale = Math.min(1, maxW/source.width, maxH/source.height);
    return {
      w: Math.max(1, Math.round(source.width*scale)),
      h: Math.max(1, Math.round(source.height*scale))
    };
  }

  function clampZoom(value){
    return Math.max(10, Math.min(400, value));
  }

  function applyZoomCss(){
    if(!preview.width || !preview.height) return;
    const scale = zoomLevel / 100;
    const cssW = Math.max(1, preview.width * scale);
    const cssH = Math.max(1, preview.height * scale);

    canvasWrap.style.width = cssW + 'px';
    canvasWrap.style.height = cssH + 'px';
    preview.style.width = cssW + 'px';
    preview.style.height = cssH + 'px';
    overlay.style.width = cssW + 'px';
    overlay.style.height = cssH + 'px';

    $('zoomRange').value = zoomLevel;
    $('zoomVal').textContent = Math.round(zoomLevel) + '%';
  }

  function setZoom(value, mode='manual', preserveView=true){
    if(!preview.width) return;

    let focusX = .5;
    let focusY = .5;
    if(preserveView && canvasWrap.offsetWidth && canvasWrap.offsetHeight){
      focusX = (
        stageArea.scrollLeft + stageArea.clientWidth / 2 - canvasWrap.offsetLeft
      ) / canvasWrap.offsetWidth;
      focusY = (
        stageArea.scrollTop + stageArea.clientHeight / 2 - canvasWrap.offsetTop
      ) / canvasWrap.offsetHeight;
      focusX = Math.max(0, Math.min(1, focusX));
      focusY = Math.max(0, Math.min(1, focusY));
    }

    zoomLevel = clampZoom(Math.round(value));
    zoomMode = mode;
    applyZoomCss();

    if(preserveView){
      requestAnimationFrame(() => {
        const targetLeft =
          canvasWrap.offsetLeft + focusX * canvasWrap.offsetWidth - stageArea.clientWidth / 2;
        const targetTop =
          canvasWrap.offsetTop + focusY * canvasWrap.offsetHeight - stageArea.clientHeight / 2;
        stageArea.scrollLeft = Math.max(0, targetLeft);
        stageArea.scrollTop = Math.max(0, targetTop);
      });
    }
  }

  function fitZoomToStage(){
    if(!preview.width || !stageArea.clientWidth || !stageArea.clientHeight) return;
    const usableW = Math.max(80, stageArea.clientWidth - 40);
    const usableH = Math.max(80, stageArea.clientHeight - 40);
    const fit = Math.min(1, usableW / preview.width, usableH / preview.height);
    zoomLevel = clampZoom(Math.floor(fit * 100));
    zoomMode = 'fit';
    applyZoomCss();
    requestAnimationFrame(() => {
      stageArea.scrollLeft = 0;
      stageArea.scrollTop = 0;
    });
  }

  async function renderPreview(){
    if(!source.width) return;
    const token = ++renderToken;
    const {w,h} = calcPreviewSize();
    preview.width = w;
    preview.height = h;
    overlay.width = w;
    overlay.height = h;

    pctx.save();
    pctx.clearRect(0,0,w,h);
    const f = filters();
    pctx.filter = `brightness(${f.brightness}%) contrast(${f.contrast}%) saturate(${f.saturation}%)`;
    pctx.drawImage(source,0,0,w,h);
    pctx.restore();

    if(f.sharpen > 0){
      const imageData = pctx.getImageData(0,0,w,h);
      applySharpen(imageData, w, h, f.sharpen);
      if(token !== renderToken) return;
      pctx.putImageData(imageData,0,0);
    }

    if(zoomMode === 'fit'){
      requestAnimationFrame(fitZoomToStage);
    }else{
      applyZoomCss();
    }

    if(compareHolding){
      renderBeforeComparison();
    }else{
      drawOverlay();
    }
  }

  function histogramStats(hist, total){
    if(!total) return {mean:128,p10:64,p50:128,p90:192};

    let sum=0;
    for(let i=0;i<256;i++) sum += hist[i] * i;

    function percentile(frac){
      const target = total * frac;
      let acc=0;
      for(let i=0;i<256;i++){
        acc += hist[i];
        if(acc >= target) return i;
      }
      return 255;
    }

    const p10=percentile(.10);
    const p50=percentile(.50);
    const p90=percentile(.90);

    let trimmedSum=0, trimmedCount=0;
    for(let i=p10;i<=p90;i++){
      trimmedSum += hist[i] * i;
      trimmedCount += hist[i];
    }

    return {
      mean: trimmedCount ? trimmedSum / trimmedCount : sum / total,
      p10,
      p50,
      p90
    };
  }

  function analyzePhotoBrightness(){
    if(!source.width || !source.height) return null;

    const maxSide=180;
    const scale=Math.min(1,maxSide/source.width,maxSide/source.height);
    const w=Math.max(1,Math.round(source.width*scale));
    const h=Math.max(1,Math.round(source.height*scale));

    const c=document.createElement('canvas');
    c.width=w;
    c.height=h;
    const ctx=c.getContext('2d',{willReadFrequently:true});
    ctx.drawImage(source,0,0,w,h);

    const data=ctx.getImageData(0,0,w,h).data;
    const overall=new Uint32Array(256);
    const center=new Uint32Array(256);
    let overallCount=0, centerCount=0;

    const x1=w*.20, x2=w*.80;
    const y1=h*.10, y2=h*.90;

    for(let y=0;y<h;y++){
      for(let x=0;x<w;x++){
        const i=(y*w+x)*4;
        if(data[i+3] < 16) continue;

        // Rec.709 相對亮度
        const lum=Math.max(0,Math.min(255,Math.round(
          data[i]*0.2126 + data[i+1]*0.7152 + data[i+2]*0.0722
        )));

        overall[lum]++;
        overallCount++;

        if(x>=x1 && x<=x2 && y>=y1 && y<=y2){
          center[lum]++;
          centerCount++;
        }
      }
    }

    const allStats=histogramStats(overall,overallCount);
    const centerStats=histogramStats(center,centerCount || overallCount);

    // 會員大頭照常有亮背景，因此中央區域權重較高，
    // 但仍混合整體亮度避免只看局部。
    const reference=centerCount
      ? centerStats.mean*.72 + allStats.mean*.28
      : allStats.mean;

    const dynamicRange=Math.max(1,allStats.p90-allStats.p10);

    return {
      reference,
      dynamicRange,
      overall:allStats,
      center:centerStats
    };
  }

  function calculateSmartAdjustments(stats){
    if(!stats){
      return {brightness:112,contrast:104,saturation:102,sharpen:1,label:'無法分析'};
    }

    const ref=stats.reference;
    let boost=Math.max(0,(145-ref)*.36);

    // 保護接近純白的高光，避免白背景或額頭反光過曝。
    if(stats.overall.p90>=245) boost*=.48;
    else if(stats.overall.p90>=235) boost*=.68;

    const brightness=Math.round(Math.max(100,Math.min(132,100+boost)));

    let contrast=102;
    if(stats.dynamicRange < 75) contrast=106;
    else if(stats.dynamicRange < 110) contrast=104;
    else if(stats.dynamicRange > 185) contrast=100;

    // 很暗時稍微增加對比；高光已強時則不再增加。
    if(ref < 95 && stats.overall.p90 < 235) contrast=Math.max(contrast,104);
    if(stats.overall.p90 >= 245) contrast=Math.min(contrast,102);

    const saturation=ref < 125 ? 102 : 100;
    const sharpen=1;

    let label='正常';
    if(ref < 90) label='偏暗';
    else if(ref < 120) label='稍暗';
    else if(ref < 150) label='正常';
    else label='偏亮';

    return {brightness,contrast,saturation,sharpen,label};
  }


  function isSkinLike(r,g,b){
    const max=Math.max(r,g,b), min=Math.min(r,g,b);
    const cb=128 - 0.168736*r - 0.331264*g + 0.5*b;
    const cr=128 + 0.5*r - 0.418688*g - 0.081312*b;

    const rgbRule =
      r>48 && g>30 && b>20 &&
      (max-min)>8 &&
      r>=g*.90 &&
      r>=b*.92;

    const ycbcrRule = cb>=74 && cb<=137 && cr>=126 && cr<=183;
    return rgbRule && ycbcrRule;
  }

  function integralRectSum(integral,w,x1,y1,x2,y2){
    const stride=w+1;
    const h=Math.floor(integral.length/stride)-1;
    x1=Math.max(0,Math.min(w,x1));
    x2=Math.max(0,Math.min(w,x2));
    y1=Math.max(0,Math.min(h,y1));
    y2=Math.max(0,Math.min(h,y2));
    return integral[y2*stride+x2]-integral[y1*stride+x2]
      -integral[y2*stride+x1]+integral[y1*stride+x1];
  }

  function buildIntegralFloat(values,w,h){
    const out=new Float64Array((w+1)*(h+1));
    const stride=w+1;
    for(let y=0;y<h;y++){
      let row=0;
      for(let x=0;x<w;x++){
        row+=values[y*w+x];
        out[(y+1)*stride+(x+1)]=out[y*stride+(x+1)]+row;
      }
    }
    return out;
  }

  function pointInEllipse(x,y,cx,cy,rx,ry){
    if(rx<=0 || ry<=0) return false;
    const nx=(x-cx)/rx;
    const ny=(y-cy)/ry;
    return nx*nx+ny*ny<=1;
  }

  function normalizedFacePoint(face,x,y){
    return {
      nx:(x-face.x)/face.w,
      ny:(y-face.y)/face.h
    };
  }

  function inFaceFeatureExclusion(face,x,y){
    const {nx,ny}=normalizedFacePoint(face,x,y);

    // V11 改為非常保守：臉部邊界、髮際線、下顎外圍先全部排除。
    if(ny<.07 || ny>.94 || nx<.08 || nx>.92) return true;

    // 眉毛 + 眼睛安全範圍加大，避免眼皮、眼尾與眉毛被誤判。
    const leftEye = nx>=.05 && nx<=.49 && ny>=.18 && ny<=.52;
    const rightEye = nx>=.51 && nx<=.95 && ny>=.18 && ny<=.52;
    if(leftEye || rightEye) return true;

    // 鼻樑下半部、鼻翼、鼻孔全部排除。
    if(nx>=.29 && nx<=.71 && ny>=.40 && ny<=.72) return true;

    // 嘴唇、嘴角與人中周邊大範圍排除。
    if(nx>=.14 && nx<=.86 && ny>=.63 && ny<=.88) return true;

    // 下顎左右邊容易碰到耳朵、頭髮、衣領。
    if(ny>.72 && (nx<.31 || nx>.69)) return true;

    return false;
  }

  function isSafeFaceSkinPoint(face,x,y){
    if(!face) return false;

    const {nx,ny}=normalizedFacePoint(face,x,y);
    const cx=face.x+face.w*.5;
    const cy=face.y+face.h*.50;

    // 先縮小臉部橢圓，徹底避開耳朵、髮際線與外輪廓。
    if(!pointInEllipse(x,y,cx,cy,face.w*.405,face.h*.465)) return false;
    if(inFaceFeatureExclusion(face,x,y)) return false;

    // V11 僅開放「安全皮膚島」：
    // 額頭中央、左右臉頰、下巴中央。寧願漏掉，也不要誤修五官。
    const forehead =
      nx>=.23 && nx<=.77 &&
      ny>=.075 && ny<=.185;

    const leftCheek =
      nx>=.10 && nx<=.35 &&
      ny>=.50 && ny<=.70;

    const rightCheek =
      nx>=.65 && nx<=.90 &&
      ny>=.50 && ny<=.70;

    const chin =
      nx>=.36 && nx<=.64 &&
      ny>=.875 && ny<=.935;

    return forehead || leftCheek || rightCheek || chin;
  }

  function estimatePrimaryFaceRegion(inputCanvas){
    if(!inputCanvas.width || !inputCanvas.height) return null;

    const maxSide=480;
    const scale=Math.min(1,maxSide/inputCanvas.width,maxSide/inputCanvas.height);
    const w=Math.max(1,Math.round(inputCanvas.width*scale));
    const h=Math.max(1,Math.round(inputCanvas.height*scale));

    const work=document.createElement('canvas');
    work.width=w; work.height=h;
    const ctx=work.getContext('2d',{willReadFrequently:true});
    ctx.drawImage(inputCanvas,0,0,w,h);
    const data=ctx.getImageData(0,0,w,h).data;

    const skin=new Uint8Array(w*h);
    for(let y=0;y<h;y++){
      for(let x=0;x<w;x++){
        const i=(y*w+x)*4;
        if(data[i+3]<32) continue;
        if(isSkinLike(data[i],data[i+1],data[i+2])){
          skin[y*w+x]=1;
        }
      }
    }

    // 先做一次簡單多數濾波，連接被五官切斷的小缺口並去除零星膚色雜點。
    const smooth=new Uint8Array(w*h);
    for(let y=1;y<h-1;y++){
      for(let x=1;x<w-1;x++){
        let count=0;
        for(let yy=-1;yy<=1;yy++){
          for(let xx=-1;xx<=1;xx++){
            count+=skin[(y+yy)*w+(x+xx)];
          }
        }
        if(count>=4) smooth[y*w+x]=1;
      }
    }

    const visited=new Uint8Array(w*h);
    const comps=[];
    const dirs=[[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[1,-1],[-1,1],[1,1]];

    for(let y=1;y<h-1;y++){
      for(let x=1;x<w-1;x++){
        const start=y*w+x;
        if(!smooth[start] || visited[start]) continue;

        const stack=[start];
        visited[start]=1;
        const pixels=[];
        let area=0,sumX=0,sumY=0;
        let minX=x,maxX=x,minY=y,maxY=y;

        while(stack.length){
          const pos=stack.pop();
          const py=Math.floor(pos/w);
          const px=pos-py*w;
          pixels.push(pos);
          area++;
          sumX+=px; sumY+=py;
          minX=Math.min(minX,px); maxX=Math.max(maxX,px);
          minY=Math.min(minY,py); maxY=Math.max(maxY,py);

          for(const [dx,dy] of dirs){
            const nx=px+dx, ny=py+dy;
            if(nx<=0 || nx>=w-1 || ny<=0 || ny>=h-1) continue;
            const np=ny*w+nx;
            if(smooth[np] && !visited[np]){
              visited[np]=1;
              stack.push(np);
            }
          }
        }

        if(area < w*h*.012) continue;
        const cx=sumX/area, cy=sumY/area;
        const bw=maxX-minX+1, bh=maxY-minY+1;

        const centerPenalty=Math.abs(cx-w*.5)/(w*.5);
        const verticalPenalty=Math.abs(cy-h*.40)/(h*.55);
        const shapePenalty=(bw/bh<.28 || bw/bh>1.25) ? .55 : 1;
        const upperBonus=cy<h*.68 ? 1 : .50;
        const score=area*(1-centerPenalty*.60)*(1-verticalPenalty*.30)*shapePenalty*upperBonus;

        comps.push({pixels,area,cx,cy,minX,maxX,minY,maxY,bw,bh,score});
      }
    }

    if(!comps.length) return null;
    comps.sort((a,b)=>b.score-a.score);
    const comp=comps[0];

    const rowMin=new Int32Array(h); rowMin.fill(w);
    const rowMax=new Int32Array(h); rowMax.fill(-1);
    const rowCount=new Int32Array(h);

    for(const pos of comp.pixels){
      const py=Math.floor(pos/w);
      const px=pos-py*w;
      rowMin[py]=Math.min(rowMin[py],px);
      rowMax[py]=Math.max(rowMax[py],px);
      rowCount[py]++;
    }

    let maxWidth=0;
    for(let y=comp.minY;y<=comp.maxY;y++){
      if(rowMax[y]>=rowMin[y]){
        maxWidth=Math.max(maxWidth,rowMax[y]-rowMin[y]+1);
      }
    }
    if(maxWidth<18) return null;

    const validRows=[];
    for(let y=comp.minY;y<=comp.maxY;y++){
      if(rowMax[y]<rowMin[y]) continue;
      const span=rowMax[y]-rowMin[y]+1;
      const density=rowCount[y]/span;
      if(span>=maxWidth*.52 && density>=.42){
        validRows.push(y);
      }
    }
    if(validRows.length<8) return null;

    let top=validRows[0];
    let bottom=validRows[validRows.length-1];

    // 找中段的列寬，以降低耳朵或脖子對寬度的影響。
    const widths=[];
    const centers=[];
    for(const y of validRows){
      const span=rowMax[y]-rowMin[y]+1;
      widths.push(span);
      centers.push((rowMin[y]+rowMax[y])/2);
    }
    widths.sort((a,b)=>a-b);
    centers.sort((a,b)=>a-b);
    let faceW=widths[Math.floor(widths.length*.72)];
    let centerX=centers[Math.floor(centers.length*.5)];

    // 臉部高度通常約為寬度的 1.12~1.38；超出多半是脖子。
    const detectedH=bottom-top+1;
    let faceH=Math.min(detectedH,faceW*1.38);
    faceH=Math.max(faceH,faceW*1.08);

    // 稍微往上延伸額頭，但不讓下方延伸到脖子。
    let faceY=Math.max(0,top-faceH*.035);
    if(faceY+faceH>h) faceH=h-faceY;

    let faceX=centerX-faceW/2;
    faceX=Math.max(0,Math.min(w-faceW,faceX));

    // 影像座標轉回原圖。
    const inv=1/scale;
    return {
      x:faceX*inv,
      y:faceY*inv,
      w:faceW*inv,
      h:faceH*inv,
      confidence:Math.min(1,comp.area/(Math.max(1,faceW*faceH))*.85),
      method:'skin-region'
    };
  }

  function analyzeFaceBlemishesOnCanvas(inputCanvas,level=1){
    const face=estimatePrimaryFaceRegion(inputCanvas);
    if(!face) return {spots:[],face:null};

    const maxSide=720;
    const scale=Math.min(1,maxSide/inputCanvas.width,maxSide/inputCanvas.height);
    const w=Math.max(1,Math.round(inputCanvas.width*scale));
    const h=Math.max(1,Math.round(inputCanvas.height*scale));

    const work=document.createElement('canvas');
    work.width=w; work.height=h;
    const ctx=work.getContext('2d',{willReadFrequently:true});
    ctx.drawImage(inputCanvas,0,0,w,h);

    const img=ctx.getImageData(0,0,w,h);
    const data=img.data;
    const count=w*h;
    const lum=new Float32Array(count);
    const reds=new Float32Array(count);
    const greens=new Float32Array(count);
    const blues=new Float32Array(count);
    const skin=new Uint8Array(count);

    for(let p=0;p<count;p++){
      const i=p*4;
      const r=data[i], g=data[i+1], b=data[i+2];
      reds[p]=r; greens[p]=g; blues[p]=b;
      lum[p]=r*.2126+g*.7152+b*.0722;
      skin[p]=isSkinLike(r,g,b) ? 1 : 0;
    }

    const intR=buildIntegralFloat(reds,w,h);
    const intG=buildIntegralFloat(greens,w,h);
    const intB=buildIntegralFloat(blues,w,h);
    const intL=buildIntegralFloat(lum,w,h);
    const intSkin=buildIntegralFloat(skin,w,h);

    const f={
      x:face.x*scale,
      y:face.y*scale,
      w:face.w*scale,
      h:face.h*scale
    };

    const sensitivity=Math.max(1,Math.min(3,level|0));
    const colorThreshold=[0,40,34,29][sensitivity];
    const lumThreshold=[0,26,22,18][sensitivity];
    const redThreshold=[0,22,18,15][sensitivity];
    const maxArea=[0,30,48,70][sensitivity];
    const maxDim=[0,12,16,20][sensitivity];
    const radius=Math.max(4,Math.round(Math.min(w,h)*.007));
    const mask=new Uint8Array(count);
    const scoreMap=new Uint8Array(count);

    const xStart=Math.max(radius+1,Math.floor(f.x));
    const xEnd=Math.min(w-radius-2,Math.ceil(f.x+f.w));
    const yStart=Math.max(radius+1,Math.floor(f.y));
    const yEnd=Math.min(h-radius-2,Math.ceil(f.y+f.h));

    for(let y=yStart;y<=yEnd;y++){
      for(let x=xStart;x<=xEnd;x++){
        if(!isSafeFaceSkinPoint(f,x,y)) continue;

        const pos=y*w+x;
        const i=pos*4;
        if(data[i+3]<32 || !skin[pos]) continue;

        const x1=x-radius, y1=y-radius, x2=x+radius+1, y2=y+radius+1;
        const area=(x2-x1)*(y2-y1);
        const meanR=integralRectSum(intR,w,x1,y1,x2,y2)/area;
        const meanG=integralRectSum(intG,w,x1,y1,x2,y2)/area;
        const meanB=integralRectSum(intB,w,x1,y1,x2,y2)/area;
        const meanL=integralRectSum(intL,w,x1,y1,x2,y2)/area;
        const skinRatio=integralRectSum(intSkin,w,x1,y1,x2,y2)/area;

        // 周圍必須以膚色為主，避免臉部輪廓、髮際線、五官邊緣。
        if(skinRatio<.80) continue;

        const r=data[i], g=data[i+1], b=data[i+2];
        const dr=r-meanR, dg=g-meanG, db=b-meanB;
        const colorDiff=Math.sqrt(dr*dr+dg*dg+db*db);
        const lumDiff=Math.abs(lum[pos]-meanL);

        // 青春痘常呈紅色，因此額外看「紅-綠」相對於局部平均的差。
        const localRedness=(r-g)-(meanR-meanG);

        if(
          colorDiff<colorThreshold &&
          lumDiff<lumThreshold &&
          localRedness<redThreshold
        ) continue;

        // 非常深的點即便是皮膚附近也可能是眉毛/眼睛邊緣，再保守排除。
        if(lum[pos]<42 && lumDiff>35) continue;

        const score=
          colorDiff*.70 +
          lumDiff*.55 +
          Math.max(0,localRedness)*.50;

        mask[pos]=1;
        scoreMap[pos]=Math.min(255,Math.round(score));
      }
    }

    const visited=new Uint8Array(count);
    const comps=[];
    const dirs=[
      [-1,-1],[0,-1],[1,-1],
      [-1,0],        [1,0],
      [-1,1],[0,1],[1,1]
    ];

    for(let y=yStart;y<=yEnd;y++){
      for(let x=xStart;x<=xEnd;x++){
        const start=y*w+x;
        if(!mask[start] || visited[start]) continue;

        const stack=[start];
        visited[start]=1;
        let area=0,sumX=0,sumY=0,sumScore=0;
        let minX=x,maxX=x,minY=y,maxY=y;

        while(stack.length){
          const pos=stack.pop();
          const py=Math.floor(pos/w);
          const px=pos-py*w;
          area++;
          sumX+=px; sumY+=py;
          sumScore+=scoreMap[pos];
          minX=Math.min(minX,px); maxX=Math.max(maxX,px);
          minY=Math.min(minY,py); maxY=Math.max(maxY,py);

          for(const [dx,dy] of dirs){
            const nx=px+dx, ny=py+dy;
            if(nx<xStart || nx>xEnd || ny<yStart || ny>yEnd) continue;
            const np=ny*w+nx;
            if(mask[np] && !visited[np]){
              visited[np]=1;
              stack.push(np);
            }
          }
        }

        const bw=maxX-minX+1, bh=maxY-minY+1;
        const longSide=Math.max(bw,bh);
        const shortSide=Math.max(1,Math.min(bw,bh));
        const aspect=longSide/shortSide;

        if(area<2 || area>maxArea) continue;
        if(longSide>maxDim || aspect>2.8) continue;

        const cx=sumX/area, cy=sumY/area;
        if(!isSafeFaceSkinPoint(f,cx,cy)) continue;

        const avgScore=sumScore/area;
        if(avgScore<colorThreshold+3) continue;

        comps.push({
          x:cx,
          y:cy,
          radius:Math.max(2.5,longSide*.74),
          score:avgScore+Math.min(12,area*.22),
          area
        });
      }
    }

    comps.sort((a,b)=>b.score-a.score);
    const maxSpots=[0,20,35,50][sensitivity];
    const selected=[];

    for(const c of comps){
      let duplicate=false;
      for(const s of selected){
        if(Math.hypot(c.x-s.x,c.y-s.y)<Math.max(c.radius,s.radius)*1.25){
          duplicate=true;
          break;
        }
      }
      if(!duplicate){
        selected.push(c);
        if(selected.length>=maxSpots) break;
      }
    }

    const inv=1/scale;
    return {
      face,
      spots:selected.map(s=>({
        x:s.x*inv,
        y:s.y*inv,
        radius:Math.max(3,s.radius*inv),
        score:s.score,
        area:s.area,
        mode:'face'
      }))
    };
  }

  function analyzePhotoDustOnCanvas(inputCanvas,level=1){
    if(!inputCanvas.width || !inputCanvas.height) return {spots:[],face:null};

    const maxSide=720;
    const scale=Math.min(1,maxSide/inputCanvas.width,maxSide/inputCanvas.height);
    const w=Math.max(1,Math.round(inputCanvas.width*scale));
    const h=Math.max(1,Math.round(inputCanvas.height*scale));

    const work=document.createElement('canvas');
    work.width=w;
    work.height=h;
    const ctx=work.getContext('2d',{willReadFrequently:true});
    ctx.drawImage(inputCanvas,0,0,w,h);

    const image=ctx.getImageData(0,0,w,h);
    const data=image.data;
    const lum=new Float32Array(w*h);
    const lumSq=new Float32Array(w*h);

    for(let p=0;p<w*h;p++){
      const i=p*4;
      const l=data[i]*.2126+data[i+1]*.7152+data[i+2]*.0722;
      lum[p]=l;
      lumSq[p]=l*l;
    }

    const intL=buildIntegralFloat(lum,w,h);
    const intL2=buildIntegralFloat(lumSq,w,h);

    const sensitivity=Math.max(1,Math.min(3,level|0));
    const diffThreshold=[0,48,40,34][sensitivity];
    const maxRingStd=[0,17,21,25][sensitivity];
    const edgeThreshold=[0,25,31,38][sensitivity];
    const maxArea=[0,24,42,65][sensitivity];
    const maxDim=[0,10,15,20][sensitivity];

    const outerR=5;
    const innerR=1;
    const mask=new Uint8Array(w*h);
    const scoreMap=new Uint8Array(w*h);

    function rectMean(integral,x1,y1,x2,y2){
      const area=Math.max(1,(x2-x1)*(y2-y1));
      return integralRectSum(integral,w,x1,y1,x2,y2)/area;
    }

    for(let y=outerR+1;y<h-outerR-1;y++){
      for(let x=outerR+1;x<w-outerR-1;x++){
        const pos=y*w+x;
        const i=pos*4;
        if(data[i+3]<32) continue;

        const ox1=x-outerR,oy1=y-outerR;
        const ox2=x+outerR+1,oy2=y+outerR+1;
        const ix1=x-innerR,iy1=y-innerR;
        const ix2=x+innerR+1,iy2=y+innerR+1;

        const outerArea=(ox2-ox1)*(oy2-oy1);
        const innerArea=(ix2-ix1)*(iy2-iy1);
        const ringArea=outerArea-innerArea;

        const outerSum=integralRectSum(intL,w,ox1,oy1,ox2,oy2);
        const innerSum=integralRectSum(intL,w,ix1,iy1,ix2,iy2);
        const ringSum=outerSum-innerSum;
        const ringMean=ringSum/Math.max(1,ringArea);

        const outerSq=integralRectSum(intL2,w,ox1,oy1,ox2,oy2);
        const innerSq=integralRectSum(intL2,w,ix1,iy1,ix2,iy2);
        const ringSq=outerSq-innerSq;
        const ringVariance=Math.max(
          0,
          ringSq/Math.max(1,ringArea)-ringMean*ringMean
        );
        const ringStd=Math.sqrt(ringVariance);

        // 周圍本身很複雜，通常是頭髮/背景、衣服/背景或五官邊緣。
        if(ringStd>maxRingStd) continue;

        const innerMean=innerSum/innerArea;
        const diff=Math.abs(innerMean-ringMean);
        if(diff<diffThreshold) continue;

        const leftMean=rectMean(intL,ox1,oy1,x-innerR,oy2);
        const rightMean=rectMean(intL,x+innerR+1,oy1,ox2,oy2);
        const topMean=rectMean(intL,ox1,oy1,ox2,y-innerR);
        const bottomMean=rectMean(intL,ox1,y+innerR+1,ox2,oy2);

        // 左右或上下兩側有明顯不同，即視為物體交界而跳過。
        if(
          Math.abs(leftMean-rightMean)>edgeThreshold ||
          Math.abs(topMean-bottomMean)>edgeThreshold
        ) continue;

        const sign=innerMean>=ringMean ? 1 : -1;
        let consistent=0,total=0;
        for(let yy=y-1;yy<=y+1;yy++){
          for(let xx=x-1;xx<=x+1;xx++){
            const d=(lum[yy*w+xx]-ringMean)*sign;
            if(d>diffThreshold*.40) consistent++;
            total++;
          }
        }
        if(consistent<Math.ceil(total*.45)) continue;

        const score=diff-ringStd*.75;
        mask[pos]=1;
        scoreMap[pos]=Math.max(0,Math.min(255,Math.round(score)));
      }
    }

    const visited=new Uint8Array(w*h);
    const comps=[];
    const dirs=[[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]];

    for(let y=1;y<h-1;y++){
      for(let x=1;x<w-1;x++){
        const start=y*w+x;
        if(!mask[start] || visited[start]) continue;

        const stack=[start];
        visited[start]=1;
        let area=0,sumX=0,sumY=0,sumScore=0;
        let minX=x,maxX=x,minY=y,maxY=y;

        while(stack.length){
          const pos=stack.pop();
          const py=Math.floor(pos/w);
          const px=pos-py*w;
          area++;sumX+=px;sumY+=py;sumScore+=scoreMap[pos];
          minX=Math.min(minX,px);maxX=Math.max(maxX,px);
          minY=Math.min(minY,py);maxY=Math.max(maxY,py);

          for(const [dx,dy] of dirs){
            const nx=px+dx,ny=py+dy;
            if(nx<=0||nx>=w-1||ny<=0||ny>=h-1) continue;
            const np=ny*w+nx;
            if(mask[np]&&!visited[np]){
              visited[np]=1;
              stack.push(np);
            }
          }
        }

        const bw=maxX-minX+1;
        const bh=maxY-minY+1;
        const longSide=Math.max(bw,bh);
        const shortSide=Math.max(1,Math.min(bw,bh));
        const aspect=longSide/shortSide;
        const compactness=area/Math.max(1,bw*bh);

        if(area<1 || area>maxArea) continue;
        if(longSide>maxDim || aspect>2.6) continue;
        if(area>=4 && compactness<.28) continue;

        comps.push({
          x:sumX/area,
          y:sumY/area,
          radius:Math.max(2.4,longSide*.75),
          score:sumScore/area,
          area
        });
      }
    }

    comps.sort((a,b)=>b.score-a.score);
    const selected=[];
    const maxSpots=[0,40,70,100][sensitivity];

    for(const c of comps){
      if(selected.some(s=>
        Math.hypot(c.x-s.x,c.y-s.y)<Math.max(c.radius,s.radius)*1.2
      )) continue;
      selected.push(c);
      if(selected.length>=maxSpots) break;
    }

    const inv=1/scale;
    return {
      face:null,
      spots:selected.map(s=>({
        x:s.x*inv,
        y:s.y*inv,
        radius:Math.max(3,s.radius*inv),
        score:s.score,
        area:s.area,
        mode:'dust'
      }))
    };
  }

  function analyzeSmartCleanOnCanvas(inputCanvas,level=1,mode='face'){
    return mode==='dust'
      ? analyzePhotoDustOnCanvas(inputCanvas,level)
      : analyzeFaceBlemishesOnCanvas(inputCanvas,level);
  }

  function healCanvasAt(canvas,ctx,x,y,radius,strength=.62){
    radius=Math.max(3,Math.round(radius));
    const left=Math.max(0,Math.floor(x-radius*2));
    const top=Math.max(0,Math.floor(y-radius*2));
    const right=Math.min(canvas.width,Math.ceil(x+radius*2));
    const bottom=Math.min(canvas.height,Math.ceil(y+radius*2));
    const w=right-left, h=bottom-top;
    if(w<4 || h<4) return false;

    const img=ctx.getImageData(left,top,w,h);
    const d=img.data;
    const ring=[];

    for(let yy=0;yy<h;yy++){
      for(let xx=0;xx<w;xx++){
        const gx=left+xx, gy=top+yy;
        const dist=Math.hypot(gx-x,gy-y);
        if(dist>=radius*1.18 && dist<=radius*1.78){
          const i=(yy*w+xx)*4;
          ring.push([d[i],d[i+1],d[i+2]]);
        }
      }
    }
    if(ring.length<8) return false;

    // 使用中位數而不是單純平均，降低周圍有髮絲或陰影時的污染。
    const rs=ring.map(v=>v[0]).sort((a,b)=>a-b);
    const gs=ring.map(v=>v[1]).sort((a,b)=>a-b);
    const bs=ring.map(v=>v[2]).sort((a,b)=>a-b);
    const mid=Math.floor(ring.length/2);
    const rr=rs[mid], gg=gs[mid], bb=bs[mid];

    for(let yy=0;yy<h;yy++){
      for(let xx=0;xx<w;xx++){
        const gx=left+xx, gy=top+yy;
        const dist=Math.hypot(gx-x,gy-y);
        if(dist<=radius){
          const i=(yy*w+xx)*4;
          const feather=Math.max(0,1-dist/radius);
          const alpha=(.18 + feather*.54)*strength;
          d[i]=d[i]*(1-alpha)+rr*alpha;
          d[i+1]=d[i+1]*(1-alpha)+gg*alpha;
          d[i+2]=d[i+2]*(1-alpha)+bb*alpha;
        }
      }
    }

    ctx.putImageData(img,left,top);
    return true;
  }

  async function processBlobSmartClean(blob,level=1,mode='face'){
    const img=await blobToImage(blob);
    const c=document.createElement('canvas');
    c.width=img.naturalWidth;
    c.height=img.naturalHeight;
    const ctx=c.getContext('2d',{willReadFrequently:true});
    ctx.drawImage(img,0,0);

    const analysis=analyzeSmartCleanOnCanvas(c,level,mode);
    const spots=analysis.spots || [];
    let applied=0;
    for(const spot of spots){
      const strength=mode==='face' ? .54 : .60;
      if(healCanvasAt(c,ctx,spot.x,spot.y,spot.radius,strength)) applied++;
    }

    const out=await canvasToBlob(c,'image/jpeg',.97);
    return {
      blob:out,
      detected:spots.length,
      applied,
      faceFound:!!analysis.face
    };
  }

  function applySharpen(imageData,w,h,level){
    if(level <= 0) return;
    const src = new Uint8ClampedArray(imageData.data);
    const dst = imageData.data;
    const amount = [0,0.35,0.65,1.0][level] || 0;
    for(let y=1;y<h-1;y++){
      for(let x=1;x<w-1;x++){
        const i=(y*w+x)*4;
        for(let c=0;c<3;c++){
          const center=src[i+c];
          const blur=(
            src[i-4+c]+src[i+4+c]+
            src[i-w*4+c]+src[i+w*4+c]
          )/4;
          dst[i+c]=Math.max(0,Math.min(255,center+(center-blur)*amount));
        }
      }
    }
  }

  function syncCropActionBar(){
    const bar=$('cropActionBar');
    if(bar) bar.hidden=!cropMode;
  }

  function drawOverlay(){
    syncCropActionBar();
    octx.clearRect(0,0,overlay.width,overlay.height);

    if(smartFaceRegion && smartAnalysisMode==='face' && !cropMode){
      const sx=preview.width/source.width;
      const sy=preview.height/source.height;
      const f=smartFaceRegion;
      octx.save();
      octx.strokeStyle='rgba(37,99,235,.78)';
      octx.lineWidth=Math.max(1,1.3/Math.max(.25,zoomLevel/100));
      octx.setLineDash([7,5]);

      if(f.landmarks && f.landmarks.length){
        const poly=MP_FACE_OVAL.map(i=>f.landmarks[i]).filter(Boolean);
        if(poly.length){
          octx.beginPath();
          octx.moveTo(poly[0].x*sx,poly[0].y*sy);
          for(let i=1;i<poly.length;i++){
            octx.lineTo(poly[i].x*sx,poly[i].y*sy);
          }
          octx.closePath();
          octx.stroke();
        }
      }else{
        octx.beginPath();
        octx.ellipse(
          (f.x+f.w*.5)*sx,
          (f.y+f.h*.5)*sy,
          f.w*.45*sx,
          f.h*.48*sy,
          0,0,Math.PI*2
        );
        octx.stroke();
      }
      octx.restore();
    }

    if(smartSpots.length && !cropMode){
      const sx=preview.width/source.width;
      const sy=preview.height/source.height;
      const avgScale=(sx+sy)/2;
      octx.save();
      octx.strokeStyle='rgba(234,88,12,.95)';
      octx.fillStyle='rgba(249,115,22,.10)';
      octx.lineWidth=Math.max(1,1.5/Math.max(.25,zoomLevel/100));
      octx.setLineDash([4,3]);
      for(const spot of smartSpots){
        const x=spot.x*sx;
        const y=spot.y*sy;
        const r=Math.max(5,spot.radius*avgScale*1.35);
        octx.beginPath();
        octx.arc(x,y,r,0,Math.PI*2);
        octx.fill();
        octx.stroke();
      }
      octx.restore();
    }

    if(cropMode && cropRect){
      const r = normalizedRect(cropRect);
      octx.save();
      octx.fillStyle='rgba(0,0,0,.45)';
      octx.fillRect(0,0,overlay.width,overlay.height);
      octx.clearRect(r.x,r.y,r.w,r.h);
      octx.strokeStyle='#ffffff';
      octx.lineWidth=2;
      octx.setLineDash([8,5]);
      octx.strokeRect(r.x,r.y,r.w,r.h);
      octx.restore();
    }

    if(healMode && healCursor && !cropMode){
      const radius = +$('brush').value;
      const scale = Math.max(.1, zoomLevel / 100);
      octx.save();
      octx.beginPath();
      octx.arc(healCursor.x, healCursor.y, radius, 0, Math.PI * 2);
      octx.strokeStyle='rgba(37,99,235,.95)';
      octx.lineWidth=1.5 / scale;
      octx.setLineDash([4 / scale, 3 / scale]);
      octx.stroke();
      octx.restore();
    }
  }

  function normalizedRect(r){
    const x=Math.min(r.x1,r.x2), y=Math.min(r.y1,r.y2);
    return {x,y,w:Math.abs(r.x2-r.x1),h:Math.abs(r.y2-r.y1)};
  }

  function rectToCropRect(r){
    return {x1:r.x, y1:r.y, x2:r.x + r.w, y2:r.y + r.h};
  }

  function getCropHandles(r){
    return [
      {name:'nw', x:r.x, y:r.y},
      {name:'n', x:r.x + r.w / 2, y:r.y},
      {name:'ne', x:r.x + r.w, y:r.y},
      {name:'e', x:r.x + r.w, y:r.y + r.h / 2},
      {name:'se', x:r.x + r.w, y:r.y + r.h},
      {name:'s', x:r.x + r.w / 2, y:r.y + r.h},
      {name:'sw', x:r.x, y:r.y + r.h},
      {name:'w', x:r.x, y:r.y + r.h / 2},
    ];
  }

  function getCropHit(point){
    if(!cropRect) return null;
    const r = normalizedRect(cropRect);
    const tolerance = 12;
    const handles = getCropHandles(r);

    for(const h of handles){
      if(Math.abs(point.x - h.x) <= tolerance && Math.abs(point.y - h.y) <= tolerance){
        return h.name;
      }
    }
    if(
      point.x >= r.x && point.x <= r.x + r.w &&
      point.y >= r.y && point.y <= r.y + r.h
    ){
      return 'move';
    }
    return null;
  }

  function getCursorFromCropHit(hit){
    return {
      nw:'nwse-resize',
      se:'nwse-resize',
      ne:'nesw-resize',
      sw:'nesw-resize',
      n:'ns-resize',
      s:'ns-resize',
      e:'ew-resize',
      w:'ew-resize',
      move:'move'
    }[hit] || 'default';
  }

  function updateCropCursor(hit){
    preview.style.cursor = cropMode ? getCursorFromCropHit(hit || cropHover) : 'default';
  }

  function getSelectedCropRatio(){
    const value = $('cropRatio').value;
    if(value === 'free') return null;
    const ratio = parseFloat(value);
    return Number.isFinite(ratio) && ratio > 0 ? ratio : null;
  }

  function fitRectToRatio(baseRect, ratio){
    if(!ratio) return baseRect;

    const cx = baseRect.x + baseRect.w / 2;
    const cy = baseRect.y + baseRect.h / 2;

    let w = baseRect.w;
    let h = w / ratio;

    if(h > baseRect.h){
      h = baseRect.h;
      w = h * ratio;
    }

    // 若目前框太小，仍確保不超過影像邊界
    const maxW = 2 * Math.min(cx, preview.width - cx);
    const maxH = 2 * Math.min(cy, preview.height - cy);
    const scale = Math.min(
      1,
      maxW > 0 ? maxW / w : 1,
      maxH > 0 ? maxH / h : 1
    );
    w *= scale;
    h *= scale;

    return {
      x: Math.max(0, Math.min(preview.width - w, cx - w / 2)),
      y: Math.max(0, Math.min(preview.height - h, cy - h / 2)),
      w,
      h
    };
  }

  function applyCropRatioToCurrent(){
    if(!cropMode || !cropRect) return;
    const ratio = getSelectedCropRatio();
    if(!ratio) return;
    const current = normalizedRect(cropRect);
    cropRect = rectToCropRect(fitRectToRatio(current, ratio));
    drawOverlay();
  }


  function clampRectToBounds(rect, maxW, maxH){
    let {x,y,w,h}=rect;
    if(w > maxW || h > maxH){
      const scale = Math.min(maxW / w, maxH / h);
      w *= scale;
      h *= scale;
    }
    x = Math.max(0, Math.min(maxW - w, x));
    y = Math.max(0, Math.min(maxH - h, y));
    return {x,y,w,h};
  }

  function sourceRectToPreviewRect(rect){
    return rectToCropRect({
      x: rect.x * (preview.width / source.width),
      y: rect.y * (preview.height / source.height),
      w: rect.w * (preview.width / source.width),
      h: rect.h * (preview.height / source.height)
    });
  }

  function enterSuggestedCropModeFromSourceRect(sourceRect, modeText){
    if(!source.width || !source.height) return;
    cropMode = true;
    cropDragMode = null;
    cropStartRect = null;
    cropHover = 'move';
    dragStart = null;
    healMode = false;
    healCursor = null;
    $('healBtn').classList.remove('active');
    $('healBtn').textContent='去污筆：關閉';
    $('cropBtn').classList.add('active');
    $('applyCropBtn').disabled=false;
    $('cancelCropBtn').disabled=false;
    preview.style.cursor='move';
    cropRect = sourceRectToPreviewRect(sourceRect);
    $('modeText').textContent=modeText;
    if(typeof activateToolTab==='function') activateToolTab('crop');
    drawOverlay();
  }

  function suggestTaiwanHeadshotCropRectForCanvas(inputCanvas){
    if(!inputCanvas || !inputCanvas.width || !inputCanvas.height) return null;

    const face = estimatePrimaryFaceRegion(inputCanvas);
    if(!face) return null;

    const faceAspect=face.h/Math.max(1,face.w);
    const faceAreaRatio=(face.w*face.h)/(inputCanvas.width*inputCanvas.height);
    const reliable =
      face.confidence>=.16 &&
      faceAspect>=.92 && faceAspect<=1.55 &&
      faceAreaRatio>=.018 && faceAreaRatio<=.78;

    // 會員卡 2 吋大頭照：3.5 × 4.5 cm，頭頂到下顎佔高度約 71%～80%。
    const ratio = 3.5 / 4.5;
    const targetHeadRatio = 0.75;
    const topHeadExpand = 0.19;
    const topMarginRatio = 0.10;
    const minFaceCoverage = 0.70;
    const maxFaceCoverage = 0.80;

    const faceTop = face.y;
    const chinY = face.y + face.h;
    const headTop = Math.max(0, faceTop - face.h * topHeadExpand);
    const headHeight = Math.max(1, chinY - headTop);

    let cropH = headHeight / targetHeadRatio;
    let cropW = cropH * ratio;

    const headCenterX = face.x + face.w / 2;
    const cropX = headCenterX - cropW / 2;
    const cropY = headTop - cropH * topMarginRatio;

    let rect = clampRectToBounds(
      {x:cropX, y:cropY, w:cropW, h:cropH},
      inputCanvas.width,
      inputCanvas.height
    );

    let effectiveHeadRatio = headHeight / rect.h;
    if(effectiveHeadRatio < minFaceCoverage || effectiveHeadRatio > maxFaceCoverage){
      const adjustedTarget = Math.min(maxFaceCoverage, Math.max(minFaceCoverage, effectiveHeadRatio));
      cropH = headHeight / adjustedTarget;
      cropW = cropH * ratio;
      rect = clampRectToBounds(
        {x:headCenterX - cropW / 2, y:headTop - cropH * topMarginRatio, w:cropW, h:cropH},
        inputCanvas.width,
        inputCanvas.height
      );
      effectiveHeadRatio = headHeight / rect.h;
    }

    return {
      rect,
      face,
      faceCoverage: effectiveHeadRatio,
      reliable
    };
  }

  function suggestTaiwanHeadshotCropRect(){
    return suggestTaiwanHeadshotCropRectForCanvas(source);
  }

  function initCropRect(){
    const ratio = getSelectedCropRatio();
    const full = {x:0, y:0, w:preview.width, h:preview.height};
    cropRect = rectToCropRect(ratio ? fitRectToRatio(full, ratio) : full);
  }

  function clampPointToPreview(point){
    return {
      x: Math.max(0, Math.min(preview.width, point.x)),
      y: Math.max(0, Math.min(preview.height, point.y))
    };
  }

  function updateCropDragFree(point){
    const minSize = 20;
    let left = cropStartRect.x;
    let top = cropStartRect.y;
    let right = cropStartRect.x + cropStartRect.w;
    let bottom = cropStartRect.y + cropStartRect.h;
    const dx = point.x - dragStart.x;
    const dy = point.y - dragStart.y;

    if(cropDragMode.includes('w')){
      left = Math.max(0, Math.min(right - minSize, cropStartRect.x + dx));
    }
    if(cropDragMode.includes('e')){
      right = Math.min(preview.width, Math.max(left + minSize, cropStartRect.x + cropStartRect.w + dx));
    }
    if(cropDragMode.includes('n')){
      top = Math.max(0, Math.min(bottom - minSize, cropStartRect.y + dy));
    }
    if(cropDragMode.includes('s')){
      bottom = Math.min(preview.height, Math.max(top + minSize, cropStartRect.y + cropStartRect.h + dy));
    }

    cropRect = rectToCropRect({
      x:left,
      y:top,
      w:right - left,
      h:bottom - top
    });
  }

  function updateCropDragFixed(point, ratio){
    const minSize = 20;
    const p = clampPointToPreview(point);
    const r = cropStartRect;

    // 四角：對角固定，依滑鼠位置維持比例。
    if(['nw','ne','se','sw'].includes(cropDragMode)){
      const anchor = {
        nw:{x:r.x + r.w, y:r.y + r.h},
        ne:{x:r.x,       y:r.y + r.h},
        se:{x:r.x,       y:r.y},
        sw:{x:r.x + r.w, y:r.y}
      }[cropDragMode];

      let rawW = Math.abs(p.x - anchor.x);
      let rawH = Math.abs(p.y - anchor.y);

      rawW = Math.max(minSize, rawW);
      rawH = Math.max(minSize, rawH);

      let w, h;
      if(rawW / rawH > ratio){
        h = rawH;
        w = h * ratio;
      }else{
        w = rawW;
        h = w / ratio;
      }

      // 依固定角落計算最大可用範圍。
      const maxW = cropDragMode.includes('w') ? anchor.x : preview.width - anchor.x;
      const maxH = cropDragMode.includes('n') ? anchor.y : preview.height - anchor.y;
      const scale = Math.min(1, maxW / w, maxH / h);
      w *= scale;
      h *= scale;

      // 最小尺寸同時需符合比例。
      const minW = Math.max(minSize, minSize * ratio);
      if(w < minW){
        w = Math.min(maxW, minW);
        h = w / ratio;
        if(h > maxH){
          h = maxH;
          w = h * ratio;
        }
      }

      let x = anchor.x;
      let y = anchor.y;
      if(cropDragMode.includes('w')) x = anchor.x - w;
      if(cropDragMode.includes('n')) y = anchor.y - h;

      cropRect = rectToCropRect({x, y, w, h});
      return;
    }

    // 左右邊：左右方向跟著滑鼠，垂直方向以中心向上下等量調整。
    if(cropDragMode === 'e' || cropDragMode === 'w'){
      const cy = r.y + r.h / 2;
      const fixedX = cropDragMode === 'e' ? r.x : r.x + r.w;
      let w = cropDragMode === 'e' ? p.x - fixedX : fixedX - p.x;
      w = Math.max(minSize, w);

      const maxWByX = cropDragMode === 'e' ? preview.width - fixedX : fixedX;
      const maxHCentered = 2 * Math.min(cy, preview.height - cy);
      const maxWByH = maxHCentered * ratio;
      w = Math.min(w, maxWByX, maxWByH);

      let h = w / ratio;
      const minH = minSize;
      if(h < minH){
        h = Math.min(maxHCentered, minH);
        w = h * ratio;
      }

      const x = cropDragMode === 'e' ? fixedX : fixedX - w;
      const y = cy - h / 2;
      cropRect = rectToCropRect({x, y, w, h});
      return;
    }

    // 上下邊：上下方向跟著滑鼠，水平方向以中心向左右等量調整。
    if(cropDragMode === 'n' || cropDragMode === 's'){
      const cx = r.x + r.w / 2;
      const fixedY = cropDragMode === 's' ? r.y : r.y + r.h;
      let h = cropDragMode === 's' ? p.y - fixedY : fixedY - p.y;
      h = Math.max(minSize, h);

      const maxHByY = cropDragMode === 's' ? preview.height - fixedY : fixedY;
      const maxWCentered = 2 * Math.min(cx, preview.width - cx);
      const maxHByW = maxWCentered / ratio;
      h = Math.min(h, maxHByY, maxHByW);

      let w = h * ratio;
      const minW = minSize;
      if(w < minW){
        w = Math.min(maxWCentered, minW);
        h = w / ratio;
      }

      const x = cx - w / 2;
      const y = cropDragMode === 's' ? fixedY : fixedY - h;
      cropRect = rectToCropRect({x, y, w, h});
    }
  }

  function updateCropDrag(point){
    if(!cropStartRect || !cropDragMode) return;

    const dx = point.x - dragStart.x;
    const dy = point.y - dragStart.y;

    if(cropDragMode === 'move'){
      const newX = Math.max(0, Math.min(preview.width - cropStartRect.w, cropStartRect.x + dx));
      const newY = Math.max(0, Math.min(preview.height - cropStartRect.h, cropStartRect.y + dy));
      cropRect = rectToCropRect({
        x:newX,
        y:newY,
        w:cropStartRect.w,
        h:cropStartRect.h
      });
      return;
    }

    const ratio = getSelectedCropRatio();
    if(ratio){
      updateCropDragFixed(point, ratio);
    }else{
      updateCropDragFree(point);
    }
  }

  function pushHistory(){
    if(!source.width) return;
    const snap = source.toDataURL('image/png');
    history = history.slice(0,historyIndex+1);
    history.push(snap);
    if(history.length > 15) history.shift();
    historyIndex = history.length-1;
    updateHistoryButtons();
  }

  async function restoreHistory(index){
    if(index<0 || index>=history.length) return;
    const img = new Image();
    img.onload=()=>{
      source.width=img.width;
      source.height=img.height;
      sctx.clearRect(0,0,source.width,source.height);
      sctx.drawImage(img,0,0);
      historyIndex=index;
      sourceDirty=true;
      touchCurrentBatchItem();
      updateMeta();
      updateHistoryButtons();
      renderPreview();
    };
    img.src=history[index];
  }

  function updateHistoryButtons(){
    $('undoBtn').disabled = !(historyIndex>0);
    $('redoBtn').disabled = !(historyIndex>=0 && historyIndex<history.length-1);
  }

  function loadFile(file){
    if(!file || !file.type.startsWith('image/')) return;
    originalName = file.name.replace(/\.[^.]+$/,'') || 'photo';
    const url=URL.createObjectURL(file);
    const img=new Image();
    img.onload=()=>{
      originalImage=img;
      compareOriginalImage=img;
      compareHolding=false;
      sourceHasTransparency=false;
      source.width=img.naturalWidth;
      source.height=img.naturalHeight;
      sctx.clearRect(0,0,source.width,source.height);
      sctx.drawImage(img,0,0);
      history=[];
      historyIndex=-1;
      zoomMode='fit';
      zoomLevel=100;
      healCursor=null;
      smartSpots=[];
      smartFaceRegion=null;
      cropDragMode=null;
      cropStartRect=null;
      cropHover=null;
      preview.style.cursor='default';
      resetFilterValues();
      resetAutoInfo();
      resetSmartCleanInfo();
      if($('bgOutputMode')) $('bgOutputMode').value='white';
      if($('bgFeather')) $('bgFeather').value='1.5';
      if($('bgFeatherVal')) $('bgFeatherVal').textContent='1.5 px';
      if($('bgRemoveInfo')) $('bgRemoveInfo').textContent=
        '使用 MediaPipe 人物分割模型保留頭髮、臉部、身體與衣服。透明背景請使用 PNG 下載。';
      pushHistory();
      sourceDirty=false;
      canvasWrap.hidden=false;
      empty.hidden=true;
      setEnabled(true);
      updateMeta();
      renderPreview();
      URL.revokeObjectURL(url);
    };
    img.src=url;
  }

  function rotate(deg){
    if(!source.width) return;
    const temp=document.createElement('canvas');
    const t=temp.getContext('2d');
    const cw=source.width, ch=source.height;
    if(Math.abs(deg)===90){
      temp.width=ch;
      temp.height=cw;
    }else{
      temp.width=cw; temp.height=ch;
    }
    t.translate(temp.width/2,temp.height/2);
    t.rotate(deg*Math.PI/180);
    t.drawImage(source,-cw/2,-ch/2);
    source.width=temp.width;
    source.height=temp.height;
    sctx.clearRect(0,0,source.width,source.height);
    sctx.drawImage(temp,0,0);
    sourceDirty=true;
    smartSpots=[];
    smartFaceRegion=null;
    $('smartApplyBtn').disabled=true;
    pushHistory();
    touchCurrentBatchItem();
    updateMeta();
    renderPreview();
  }

  function applyCrop(){
    if(!cropRect) return;
    const r=normalizedRect(cropRect);
    if(r.w<10 || r.h<10) return;
    const sx=source.width/preview.width;
    const sy=source.height/preview.height;
    const x=Math.round(r.x*sx);
    const y=Math.round(r.y*sy);
    const w=Math.max(1,Math.round(r.w*sx));
    const h=Math.max(1,Math.round(r.h*sy));

    const temp=document.createElement('canvas');
    temp.width=w; temp.height=h;
    temp.getContext('2d').drawImage(source,x,y,w,h,0,0,w,h);
    source.width=w; source.height=h;
    sctx.drawImage(temp,0,0);

    cropRect=null;
    cropMode=false;
    cropDragMode=null;
    cropStartRect=null;
    cropHover=null;
    preview.style.cursor='default';
    $('cropBtn').classList.remove('active');
    $('applyCropBtn').disabled=true;
    $('cancelCropBtn').disabled=true;
    $('modeText').textContent=editorMode==='batch'
      ? `批次處理：第 ${batchIndex+1} / ${batchItems.length} 張`
      : '預覽';
    sourceDirty=true;
    smartSpots=[];
    smartFaceRegion=null;
    $('smartApplyBtn').disabled=true;
    pushHistory();
    touchCurrentBatchItem();
    updateMeta();
    renderPreview();
  }

  function cancelCrop(){
    cropRect=null;
    cropMode=false;
    dragStart=null;
    cropDragMode=null;
    cropStartRect=null;
    cropHover=null;
    preview.style.cursor='default';
    $('cropBtn').classList.remove('active');
    $('applyCropBtn').disabled=true;
    $('cancelCropBtn').disabled=true;
    $('modeText').textContent = healMode
      ? '去污筆'
      : (editorMode==='batch' && batchIndex>=0
        ? `批次處理：第 ${batchIndex+1} / ${batchItems.length} 張`
        : '預覽');
    drawOverlay();
  }

  function canvasPoint(ev){
    const rect=preview.getBoundingClientRect();
    return {
      x:(ev.clientX-rect.left)*(preview.width/rect.width),
      y:(ev.clientY-rect.top)*(preview.height/rect.height)
    };
  }

  function healAt(px,py){
    if(!source.width) return;
    const scaleX=source.width/preview.width;
    const scaleY=source.height/preview.height;
    const x=Math.round(px*scaleX);
    const y=Math.round(py*scaleY);
    const radius=Math.max(4,Math.round(+$('brush').value*((scaleX+scaleY)/2)));

    const left=Math.max(0,x-radius*2);
    const top=Math.max(0,y-radius*2);
    const right=Math.min(source.width,x+radius*2);
    const bottom=Math.min(source.height,y+radius*2);
    const w=right-left, h=bottom-top;
    if(w<3 || h<3) return;

    const img=sctx.getImageData(left,top,w,h);
    const d=img.data;
    let sr=0,sg=0,sb=0,count=0;

    for(let yy=0;yy<h;yy++){
      for(let xx=0;xx<w;xx++){
        const gx=left+xx, gy=top+yy;
        const dist=Math.hypot(gx-x,gy-y);
        if(dist>=radius*1.15 && dist<=radius*1.8){
          const i=(yy*w+xx)*4;
          sr+=d[i]; sg+=d[i+1]; sb+=d[i+2]; count++;
        }
      }
    }
    if(!count) return;
    const ar=sr/count, ag=sg/count, ab=sb/count;

    for(let yy=0;yy<h;yy++){
      for(let xx=0;xx<w;xx++){
        const gx=left+xx, gy=top+yy;
        const dist=Math.hypot(gx-x,gy-y);
        if(dist<=radius){
          const i=(yy*w+xx)*4;
          const edge=Math.max(0,Math.min(1,1-dist/radius));
          const alpha=.28 + edge*.48;
          d[i]=d[i]*(1-alpha)+ar*alpha;
          d[i+1]=d[i+1]*(1-alpha)+ag*alpha;
          d[i+2]=d[i+2]*(1-alpha)+ab*alpha;
        }
      }
    }
    sctx.putImageData(img,left,top);
    sourceDirty=true;
    smartSpots=[];
    smartFaceRegion=null;
    $('smartApplyBtn').disabled=true;
    pushHistory();
    touchCurrentBatchItem();
    renderPreview();
  }

  async function exportCanvas(type='image/png'){
    const f=filters();

    let outW=source.width, outH=source.height;
    const preset=$('presetSize').value;
    if(preset!=='original'){
      outW=Math.max(1,parseInt($('outW').value)||source.width);
      outH=Math.max(1,parseInt($('outH').value)||source.height);
    }

    const base=document.createElement('canvas');
    base.width=outW; base.height=outH;
    const b=base.getContext('2d',{willReadFrequently:true});

    if(type==='image/jpeg' && sourceHasTransparency){
      b.fillStyle='#ffffff';
      b.fillRect(0,0,outW,outH);
    }

    b.filter=`brightness(${f.brightness}%) contrast(${f.contrast}%) saturate(${f.saturation}%)`;

    if($('keepRatio').checked && preset!=='original'){
      const scale=Math.min(outW/source.width,outH/source.height);
      const dw=Math.round(source.width*scale);
      const dh=Math.round(source.height*scale);
      const dx=Math.round((outW-dw)/2);
      const dy=Math.round((outH-dh)/2);
      if(!sourceHasTransparency || type==='image/jpeg'){
        b.fillStyle='#ffffff';
        b.fillRect(0,0,outW,outH);
      }
      b.drawImage(source,dx,dy,dw,dh);
    }else{
      b.drawImage(source,0,0,outW,outH);
    }
    b.filter='none';

    if(f.sharpen>0){
      const data=b.getImageData(0,0,outW,outH);
      applySharpen(data,outW,outH,f.sharpen);
      b.putImageData(data,0,0);
    }
    return base;
  }

  async function download(type){
    const c=await exportCanvas(type);
    const ext=type==='image/png'?'png':'jpg';
    const quality=+$('quality').value/100;
    c.toBlob(blob=>{
      if(!blob) return;
      const a=document.createElement('a');
      a.href=URL.createObjectURL(blob);
      a.download=`${originalName}_edited.${ext}`;
      document.body.appendChild(a);
      a.click();
      setTimeout(()=>{
        URL.revokeObjectURL(a.href);
        a.remove();
      },1000);
    },type,quality);
  }

  $('singleModeBtn').onclick=()=>switchEditorMode('single');
  $('batchModeBtn').onclick=()=>switchEditorMode('batch');

  $('openBtn').onclick=()=>{
    if(editorMode==='batch') batchFileInput.click();
    else fileInput.click();
  };
  $('batchAddBtn').onclick=()=>batchFileInput.click();

  fileInput.onchange=e=>{
    const file=e.target.files[0];
    if(file) loadFile(file);
    fileInput.value='';
  };

  batchFileInput.onchange=async e=>{
    await addBatchFiles(e.target.files);
    batchFileInput.value='';
  };

  dropZone.addEventListener('dragover',e=>{
    e.preventDefault(); app.classList.add('drop-active');
  });
  dropZone.addEventListener('dragleave',()=>app.classList.remove('drop-active'));
  dropZone.addEventListener('drop',async e=>{
    e.preventDefault(); app.classList.remove('drop-active');
    const files=[...e.dataTransfer.files].filter(f=>f.type.startsWith('image/'));
    if(editorMode==='batch'){
      await addBatchFiles(files);
    }else if(files[0]){
      loadFile(files[0]);
    }
  });

  $('batchPrevBtn').onclick=()=>loadBatchItem(batchIndex-1);
  $('batchNextBtn').onclick=()=>loadBatchItem(batchIndex+1);
  $('batchCropBtn').onclick=runBatchAutoHeadshotCropMP;
  $('batchAutoBtn').onclick=runBatchSmartBright;
  $('batchCleanBtn').onclick=runBatchSmartCleanMP;
  $('batchBgWhiteBtn').onclick=()=>runBatchRemoveBackgroundMP('white');
  $('batchBgTransparentBtn').onclick=()=>runBatchRemoveBackgroundMP('transparent');
  $('batchZipBtn').onclick=exportBatchZip;
  $('batchClearBtn').onclick=clearBatchItems;

  $('batchDoneBtn').onclick=async()=>{
    const item=batchItems[batchIndex];
    if(!item) return;
    await saveCurrentBatchItem();
    item.done=!item.done;
    if(!item.done) item.adjusted=true;
    renderBatchList();
  };

  $('rotateLeftBtn').onclick=()=>rotate(-90);
  $('rotateRightBtn').onclick=()=>rotate(90);


  $('autoHeadshotCropBtn').onclick=async()=>{
    if(!source.width) return;

    $('cropRatio').value='0.7777777778';
    $('modeText').textContent='MediaPipe 正在定位臉部與頭頂…';

    try{
      const suggestion=await suggestTaiwanHeadshotCropRectMP(source);
      if(!suggestion || !suggestion.reliable){
        $('modeText').textContent='MediaPipe 未能可靠找到臉部，為避免錯裁已停止。請改用手動裁切。';
        return;
      }

      const coveragePct=Math.round(suggestion.faceCoverage*100);
      enterSuggestedCropModeFromSourceRect(
        suggestion.rect,
        `MediaPipe 2 吋裁切建議：頭頂至下顎約佔 ${coveragePct}%（目標 70%～80%）。可微調後套用。`
      );
    }catch(err){
      console.error(err);
      $('modeText').textContent='MediaPipe 自動裁切失敗，請確認網路連線後重試。';
    }
  };

  $('cropBtn').onclick=()=>{
    if(!source.width) return;
    smartSpots=[];
    smartFaceRegion=null;
    $('smartApplyBtn').disabled=true;
    healMode=false;
    healCursor=null;
    $('healBtn').classList.remove('active');
    $('healBtn').textContent='去污筆：關閉';
    cropMode=!cropMode;

    if(cropMode){
      if(typeof activateToolTab==='function') activateToolTab('crop');
      initCropRect();
      cropDragMode=null;
      cropStartRect=null;
      cropHover='move';
      updateCropCursor('move');
      $('cropBtn').classList.add('active');
      $('applyCropBtn').disabled=false;
      $('cancelCropBtn').disabled=false;
      const ratioLabel = $('cropRatio').options[$('cropRatio').selectedIndex].text;
      $('modeText').textContent=`裁切模式（${ratioLabel}）：拖曳四角、上下左右手把，或拖曳框內移動`;
    }else{
      cropRect=null;
      cropDragMode=null;
      cropStartRect=null;
      cropHover=null;
      preview.style.cursor='default';
      $('cropBtn').classList.remove('active');
      $('applyCropBtn').disabled=true;
      $('cancelCropBtn').disabled=true;
      $('modeText').textContent='預覽';
    }
    drawOverlay();
  };
  $('applyCropBtn').onclick=applyCrop;
  $('cancelCropBtn').onclick=cancelCrop;

  $('cropRatio').addEventListener('change',()=>{
    if(cropMode){
      const ratio = getSelectedCropRatio();
      if(ratio){
        applyCropRatioToCurrent();
      }
      const ratioLabel = $('cropRatio').options[$('cropRatio').selectedIndex].text;
      $('modeText').textContent=`裁切模式（${ratioLabel}）：拖曳四角、上下左右手把，或拖曳框內移動`;
    }
  });

  preview.addEventListener('pointerdown',ev=>{
    if(cropMode){
      const p=canvasPoint(ev);
      const hit=getCropHit(p);
      if(!hit) return;
      preview.setPointerCapture(ev.pointerId);
      dragStart=p;
      cropDragMode=hit;
      cropStartRect=normalizedRect(cropRect);
      cropHover=hit;
      updateCropCursor(hit);
      drawOverlay();
    }else if(healMode){
      const p=canvasPoint(ev);
      healAt(p.x,p.y);
    }
  });
  preview.addEventListener('pointermove',ev=>{
    const p=canvasPoint(ev);
    if(cropMode && dragStart && cropDragMode){
      updateCropDrag(p);
      drawOverlay();
    }else if(cropMode){
      cropHover=getCropHit(p);
      updateCropCursor();
    }else if(healMode){
      healCursor=p;
      drawOverlay();
    }
  });

  preview.addEventListener('pointerleave',()=>{
    if(cropMode && !dragStart){
      cropHover=null;
      updateCropCursor();
    }
    if(healMode){
      healCursor=null;
      drawOverlay();
    }
  });
  preview.addEventListener('pointerup',ev=>{
    if(cropMode && dragStart){
      const p=canvasPoint(ev);
      updateCropDrag(p);
      dragStart=null;
      cropStartRect=null;
      cropDragMode=null;
      cropHover=getCropHit(p);
      updateCropCursor();
      const r=normalizedRect(cropRect);
      $('applyCropBtn').disabled=!(r.w>=10 && r.h>=10);
      drawOverlay();
    }
  });

  ['brightness','contrast','saturation','sharpen'].forEach(id=>{
    $(id).addEventListener('input',()=>{
      syncLabels();
      touchCurrentBatchItem();
      renderPreview();
    });
  });

  $('brush').oninput=()=>{
    syncLabels();
    drawOverlay();
  };
  $('quality').oninput=syncLabels;

  $('autoBtn').onclick=()=>{
    const stats=analyzePhotoBrightness();
    const adj=calculateSmartAdjustments(stats);

    $('brightness').value=adj.brightness;
    $('contrast').value=adj.contrast;
    $('saturation').value=adj.saturation;
    $('sharpen').value=adj.sharpen;

    syncLabels();
    touchCurrentBatchItem();
    renderPreview();

    if(stats){
      const refPct=Math.round(stats.reference/255*100);
      $('autoInfo').textContent=
        `分析結果：${adj.label}（基準亮度約 ${refPct}%）。`+
        `已套用：亮度 ${adj.brightness}%、對比 ${adj.contrast}%、`+
        `飽和度 ${adj.saturation}%、銳化 ${adj.sharpen}。`;
    }
  };

  $('quickBrightBtn').onclick=()=>{
    $('brightness').value=112;
    $('contrast').value=104;
    $('saturation').value=102;
    $('sharpen').value=1;
    syncLabels();
    renderPreview();
    $('autoInfo').textContent=
      '已使用快速提亮：固定套用亮度 112%、對比 104%、飽和度 102%、銳化 1。';
    touchCurrentBatchItem();
  };

  $('resetFilterBtn').onclick=()=>{
    resetFilterValues();
    resetAutoInfo();
    touchCurrentBatchItem();
    renderPreview();
  };


  $('compareHoldBtn').addEventListener('pointerdown',ev=>{
    if($('compareHoldBtn').disabled) return;
    ev.preventDefault();
    try{$('compareHoldBtn').setPointerCapture(ev.pointerId);}catch{}
    startCompareBefore();
  });

  $('compareHoldBtn').addEventListener('pointerup',ev=>{
    ev.preventDefault();
    endCompareBefore();
  });
  $('compareHoldBtn').addEventListener('pointercancel',endCompareBefore);
  $('compareHoldBtn').addEventListener('lostpointercapture',endCompareBefore);

  $('compareHoldBtn').addEventListener('keydown',ev=>{
    if((ev.key===' ' || ev.key==='Enter') && !compareHolding){
      ev.preventDefault();
      startCompareBefore();
    }
  });
  $('compareHoldBtn').addEventListener('keyup',ev=>{
    if(ev.key===' ' || ev.key==='Enter'){
      ev.preventDefault();
      endCompareBefore();
    }
  });

  $('bgFeather').addEventListener('input',()=>{
    $('bgFeatherVal').textContent=`${$('bgFeather').value} px`;
  });

  $('removeBgBtn').onclick=async()=>{
    if(!source.width) return;
    await removePersonBackground(
      $('bgOutputMode').value,
      +$('bgFeather').value
    );
  };

  $('smartCleanMode').addEventListener('change',()=>{
    smartSpots=[];
    smartFaceRegion=null;
    smartAnalysisMode=$('smartCleanMode').value;
    $('smartApplyBtn').disabled=true;
    $('smartCleanInfo').textContent=smartAnalysisMode==='face'
      ? '已切換為「臉部斑點／青春痘」：MediaPipe 會先取得 Face Landmark，再排除眼睛、眉毛、鼻子、嘴巴與臉部外輪廓。請按「分析污點」。'
      : '已切換為「照片灰塵／小型損傷」：採孤立污點＋邊緣保護，會排除頭髮、衣服與背景等高對比交界。請按「分析污點」。';
    drawOverlay();
  });

  $('smartCleanLevel').addEventListener('input',()=>{
    syncLabels();
    if(smartSpots.length || smartFaceRegion){
      smartSpots=[];
      smartFaceRegion=null;
      $('smartApplyBtn').disabled=true;
      $('smartCleanInfo').textContent='偵測強度已變更，請重新按「分析污點」。';
      drawOverlay();
    }
  });

  $('smartAnalyzeBtn').onclick=async()=>{
    if(!source.width) return;

    if(cropMode) cancelCrop();
    healMode=false;
    healCursor=null;
    $('healBtn').classList.remove('active');
    $('healBtn').textContent='去污筆：關閉';

    const level=+$('smartCleanLevel').value;
    const mode=$('smartCleanMode').value;
    smartAnalysisMode=mode;
    smartSpots=[];
    smartFaceRegion=null;
    $('smartApplyBtn').disabled=true;

    if(mode==='dust'){
      $('smartCleanInfo').textContent='正在分析孤立灰塵／小型損傷，並排除高對比物體邊界…';
      await new Promise(r=>setTimeout(r,20));
      const result=analyzePhotoDustOnCanvas(source,level);
      smartSpots=result.spots || [];
      smartFaceRegion=null;
      $('smartApplyBtn').disabled=smartSpots.length===0;
      const label=smartCleanLevelLabel();
      $('smartCleanInfo').textContent=smartSpots.length
        ? `照片損傷分析完成（${label}）：找到 ${smartSpots.length} 個疑似灰塵／小黑白點。請確認橘色標記後再套用。`
        : `照片損傷分析完成（${label}）：沒有找到符合條件的小型黑白污點。`;
      drawOverlay();
      return;
    }

    $('smartCleanInfo').textContent='MediaPipe 正在定位臉部 Landmark，建立五官排除區與安全皮膚遮罩…';

    try{
      const result=await analyzeFaceBlemishesMediaPipe(source,level);
      smartSpots=result.spots || [];
      smartFaceRegion=result.face || null;
      $('smartApplyBtn').disabled=smartSpots.length===0;
      const label=smartCleanLevelLabel();

      if(!smartFaceRegion){
        $('smartCleanInfo').textContent=
          'MediaPipe 沒有辨識到可靠臉部，因此沒有自動標記，避免誤修頭髮、衣服或背景。';
      }else if(smartSpots.length){
        $('smartCleanInfo').textContent=
          `MediaPipe 臉部分析完成（${label}）：找到 ${smartSpots.length} 個安全皮膚區候選斑點。`+
          ' 已依 Landmark 排除眼睛、眉毛、鼻子、嘴巴與臉部外輪廓。';
      }else{
        $('smartCleanInfo').textContent=
          `MediaPipe 已定位臉部（${label}），但安全皮膚區內沒有符合條件的小型斑點。`+
          '可提高偵測強度，或使用手動去污筆。';
      }
      drawOverlay();
    }catch(err){
      console.error(err);
      smartSpots=[];
      smartFaceRegion=null;
      $('smartApplyBtn').disabled=true;
      $('smartCleanInfo').textContent='MediaPipe 臉部分析失敗，請確認網路連線後重試；手動去污筆仍可使用。';
      drawOverlay();
    }
  };

  $('smartApplyBtn').onclick=()=>{
    if(!source.width || !smartSpots.length) return;

    const count=smartSpots.length;
    const mode=smartAnalysisMode;
    let applied=0;
    for(const spot of smartSpots){
      const strength=mode==='face' ? .54 : .60;
      if(healCanvasAt(source,sctx,spot.x,spot.y,spot.radius,strength)) applied++;
    }

    smartSpots=[];
    smartFaceRegion=null;
    $('smartApplyBtn').disabled=true;
    sourceDirty=true;
    pushHistory();
    touchCurrentBatchItem();
    $('smartCleanInfo').textContent=mode==='face'
      ? `MediaPipe 臉部智慧去污已完成：原本標記 ${count} 個安全皮膚位置，已保守修補 ${applied} 個。建議放大 200%～400% 檢查。`
      : `照片灰塵／小型損傷修復已完成：原本標記 ${count} 個位置，已修補 ${applied} 個。建議放大檢查。`;
    renderPreview();
  };

  $('smartClearBtn').onclick=()=>{
    smartSpots=[];
    smartFaceRegion=null;
    $('smartApplyBtn').disabled=true;
    $('smartCleanInfo').textContent='已清除分析標記，照片內容沒有被修改。';
    drawOverlay();
  };

  $('healBtn').onclick=()=>{
    if(!source.width) return;
    smartSpots=[];
    smartFaceRegion=null;
    $('smartApplyBtn').disabled=true;
    cropMode=false;
    cropRect=null;
    dragStart=null;
    $('cropBtn').classList.remove('active');
    $('applyCropBtn').disabled=true;
    $('cancelCropBtn').disabled=true;
    healMode=!healMode;
    healCursor=null;
    $('healBtn').classList.toggle('active',healMode);
    $('healBtn').textContent=healMode?'去污筆：開啟':'去污筆：關閉';
    $('modeText').textContent=healMode?'去污筆：可先放大至 200%～400% 細修':'預覽';
    drawOverlay();
  };

  $('zoomRange').addEventListener('input',()=>{
    setZoom(+$('zoomRange').value, 'manual', true);
  });
  $('zoomOutBtn').onclick=()=>setZoom(zoomLevel-25, 'manual', true);
  $('zoomInBtn').onclick=()=>setZoom(zoomLevel+25, 'manual', true);
  $('fitZoomBtn').onclick=fitZoomToStage;

  stageArea.addEventListener('wheel',ev=>{
    if(ev.ctrlKey && source.width){
      ev.preventDefault();
      setZoom(zoomLevel + (ev.deltaY < 0 ? 25 : -25), 'manual', true);
    }
  }, {passive:false});

  window.addEventListener('resize',()=>{
    if(zoomMode === 'fit' && source.width){
      requestAnimationFrame(fitZoomToStage);
    }
  });

  $('undoBtn').onclick=()=>restoreHistory(historyIndex-1);
  $('redoBtn').onclick=()=>restoreHistory(historyIndex+1);

  $('resetAllBtn').onclick=async()=>{
    if(editorMode==='batch'){
      const item=batchItems[batchIndex];
      if(!item) return;
      item.editedBlob=null;
      item.filters=defaultBatchFilters();
      item.autoInfo='';
      item.adjusted=false;
      item.done=false;
      item.autoCropped=false;
      item.hasTransparency=false;
      sourceHasTransparency=false;
      sourceDirty=false;
      await loadBlobIntoEditor(item.file,item.file.name,item.filters,item.autoInfo,item.file);
      resetSmartCleanInfo();
      $('modeText').textContent=`批次處理：第 ${batchIndex+1} / ${batchItems.length} 張`;
      renderBatchList();
      return;
    }

    if(!originalImage) return;
    source.width=originalImage.naturalWidth;
    source.height=originalImage.naturalHeight;
    sctx.clearRect(0,0,source.width,source.height);
    sctx.drawImage(originalImage,0,0);
    history=[];
    historyIndex=-1;
    zoomMode='fit';
    zoomLevel=100;
    healCursor=null;
    cropDragMode=null;
    cropStartRect=null;
    cropHover=null;
    preview.style.cursor='default';
    resetFilterValues();
    resetAutoInfo();
    resetSmartCleanInfo();
    pushHistory();
    sourceDirty=false;
    cancelCrop();
    healMode=false;
    $('healBtn').classList.remove('active');
    $('healBtn').textContent='去污筆：關閉';
    updateMeta();
    renderPreview();
  };

  $('presetSize').onchange=()=>{
    const v=$('presetSize').value;
    const custom=v==='custom';
    $('outW').disabled=!custom;
    $('outH').disabled=!custom;
    if(v==='original'){
      $('outW').value=source.width||'';
      $('outH').value=source.height||'';
    }else if(!custom){
      const [w,h]=v.split('x');
      $('outW').value=w;
      $('outH').value=h;
    }else{
      $('outW').disabled=false;
      $('outH').disabled=false;
    }
  };

  $('outW').addEventListener('input',()=>{
    if(!$('keepRatio').checked || !source.width) return;
    const w=parseInt($('outW').value);
    if(w>0) $('outH').value=Math.max(1,Math.round(w*source.height/source.width));
  });
  $('outH').addEventListener('input',()=>{
    if(!$('keepRatio').checked || !source.width) return;
    const h=parseInt($('outH').value);
    if(h>0) $('outW').value=Math.max(1,Math.round(h*source.width/source.height));
  });

  $('downloadJpgBtn').onclick=()=>download('image/jpeg');
  $('downloadPngBtn').onclick=()=>download('image/png');

  document.addEventListener('keydown',ev=>{
    if(editorMode!=='batch' || batchBusy || cropMode || healMode) return;
    const tag=(document.activeElement && document.activeElement.tagName || '').toLowerCase();
    if(['input','select','textarea'].includes(tag)) return;

    if(ev.key==='ArrowLeft' && batchIndex>0){
      ev.preventDefault();
      loadBatchItem(batchIndex-1);
    }else if(ev.key==='ArrowRight' && batchIndex<batchItems.length-1){
      ev.preventDefault();
      loadBatchItem(batchIndex+1);
    }else if(ev.key==='Enter' && batchIndex>=0){
      ev.preventDefault();
      const item=batchItems[batchIndex];
      if(item){
        item.done=true;
        saveCurrentBatchItem().then(renderBatchList);
      }
    }
  });

  // ============================================================
  // V12.3 Office-style Ribbon tabs + fixed workspace
  // ============================================================
  function activateRibbonTab(name){
    document.querySelectorAll('.ribbon-tab').forEach(btn=>{
      btn.classList.toggle('active',btn.dataset.ribbonTab===name);
    });
    document.querySelectorAll('.ribbon-panel').forEach(panel=>{
      panel.classList.toggle('active',panel.dataset.ribbonPanel===name);
    });

    app.classList.toggle('batch-ribbon-expanded',name==='batch');

    requestAnimationFrame(()=>{
      if(source.width && zoomMode==='fit'){
        fitZoomToStage();
      }
    });
  }

  document.querySelectorAll('.ribbon-tab').forEach(btn=>{
    btn.addEventListener('click',()=>activateRibbonTab(btn.dataset.ribbonTab));
  });

  function activateToolTab(name){
    document.querySelectorAll('.tool-tab').forEach(btn=>{
      btn.classList.toggle('active',btn.dataset.toolTab===name);
    });
    document.querySelectorAll('.tool-pane').forEach(pane=>{
      pane.classList.toggle('active',pane.dataset.toolPane===name);
    });
  }

  document.querySelectorAll('.tool-tab').forEach(btn=>{
    btn.addEventListener('click',()=>activateToolTab(btn.dataset.toolTab));
  });

  const toggleLeftBtn=$('toggleLeftBtn');
  const toggleRightBtn=$('toggleRightBtn');

  function refreshWorkspaceAfterPanelToggle(){
    requestAnimationFrame(()=>{
      if(source.width && zoomMode==='fit') fitZoomToStage();
      else if(source.width) applyZoomCss();
    });
  }

  if(toggleLeftBtn){
    toggleLeftBtn.addEventListener('click',()=>{
      const collapsed=app.classList.toggle('left-collapsed');
      toggleLeftBtn.classList.toggle('active',collapsed);
      toggleLeftBtn.textContent=collapsed ? '☰ 顯示照片欄' : '☰ 照片欄';
      refreshWorkspaceAfterPanelToggle();
    });
  }

  if(toggleRightBtn){
    toggleRightBtn.addEventListener('click',()=>{
      const collapsed=app.classList.toggle('right-collapsed');
      toggleRightBtn.classList.toggle('active',collapsed);
      toggleRightBtn.textContent=collapsed ? '⚙ 顯示工具欄' : '⚙ 工具欄';
      refreshWorkspaceAfterPanelToggle();
    });
  }

  if($('singleInfoPanel')) $('singleInfoPanel').hidden=false;
  app.classList.toggle('batch-active',editorMode==='batch');
  activateRibbonTab(editorMode==='batch' ? 'batch' : 'home');

  renderBatchList();
  syncLabels();

  const modelStatusBox=document.getElementById("faceModelStatus");
  if(modelStatusBox){
    modelStatusBox.style.cursor="pointer";
    modelStatusBox.title="點一下可重新載入 MediaPipe 模型";
    modelStatusBox.addEventListener("click",async()=>{
      if(mpFaceReady) return;
      mpFacePromise=null;
      mpModulePromise=null;
      try{
        await initMediaPipeFaceLandmarker();
      }catch{}
    });
  }

  // 頁面載入後預先初始化；失敗時會直接顯示錯誤，
  // 不會再永久停留在「載入中」。
  initMediaPipeFaceLandmarker().catch(()=>{});
})();
