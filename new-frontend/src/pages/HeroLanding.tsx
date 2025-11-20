// import { useEffect } from "react";
// import { useNavigate } from "react-router-dom";
// import { ChevronDown, Sparkles } from "lucide-react";
// import { motion } from "framer-motion";

// const HeroLanding = () => {
//   const navigate = useNavigate();

//   useEffect(() => {
//     const timer = setTimeout(() => {
//       navigate("/brand-selection");
//     }, 4000);

//     return () => clearTimeout(timer);
//   }, [navigate]);

//   const handleScroll = () => {
//     navigate("/brand-selection");
//   };

//   return (
//     <div className="min-h-screen bg-gradient-hero relative overflow-hidden flex items-center justify-center">
//       {/* Animated background particles - Silver & Chrome */}
//       <div className="absolute inset-0">
//         <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-secondary/20 rounded-full blur-3xl animate-float" />
//         <div className="absolute bottom-1/3 right-1/4 w-96 h-96 bg-chrome/20 rounded-full blur-3xl animate-float" style={{ animationDelay: "1s" }} />
//         <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-accent-mechanic/15 rounded-full blur-3xl animate-float" style={{ animationDelay: "2s" }} />
//       </div>

//       {/* Flowing animated lines */}
//       <div className="absolute inset-0 opacity-20">
//         <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
//           <motion.path
//             d="M 0 300 Q 400 200, 800 300 T 1600 300"
//             stroke="url(#gradient1)"
//             strokeWidth="2"
//             fill="none"
//             initial={{ pathLength: 0 }}
//             animate={{ pathLength: 1 }}
//             transition={{ duration: 3, ease: "easeInOut", repeat: Infinity }}
//           />
//           <motion.path
//             d="M 0 400 Q 400 300, 800 400 T 1600 400"
//             stroke="url(#gradient2)"
//             strokeWidth="2"
//             fill="none"
//             initial={{ pathLength: 0 }}
//             animate={{ pathLength: 1 }}
//             transition={{ duration: 3, ease: "easeInOut", repeat: Infinity, delay: 0.5 }}
//           />
//           <defs>
//             <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="0%">
//               <stop offset="0%" stopColor="hsl(210 100% 50%)" stopOpacity="0" />
//               <stop offset="50%" stopColor="hsl(210 100% 50%)" stopOpacity="1" />
//               <stop offset="100%" stopColor="hsl(210 100% 50%)" stopOpacity="0" />
//             </linearGradient>
//             <linearGradient id="gradient2" x1="0%" y1="0%" x2="100%" y2="0%">
//               <stop offset="0%" stopColor="hsl(38 100% 50%)" stopOpacity="0" />
//               <stop offset="50%" stopColor="hsl(38 100% 50%)" stopOpacity="1" />
//               <stop offset="100%" stopColor="hsl(38 100% 50%)" stopOpacity="0" />
//             </linearGradient>
//           </defs>
//         </svg>
//       </div>

//       {/* Main content */}
//       <div className="relative z-10 text-center px-4">
//         <motion.div
//           initial={{ opacity: 0, y: 30 }}
//           animate={{ opacity: 1, y: 0 }}
//           transition={{ duration: 1, ease: "easeOut" }}
//           className="mb-8"
//         >
//           <div className="inline-flex items-center gap-2 px-6 py-3 rounded-full glass-card mb-8 border border-secondary/20">
//             <Sparkles className="w-5 h-5 text-secondary animate-pulse" />
//             <span className="text-sm font-medium uppercase tracking-wider text-muted-foreground">AI-Powered Automotive Intelligence</span>
//           </div>
//         </motion.div>

//         <motion.h1
//           initial={{ opacity: 0, y: 30 }}
//           animate={{ opacity: 1, y: 0 }}
//           transition={{ duration: 1, delay: 0.3, ease: "easeOut" }}
//           className="text-6xl md:text-8xl font-bold mb-6 leading-tight"
//         >
//           <span className="text-foreground">
//             THE PINNACLE OF
//           </span>
//           <br />
//           <span className="text-gradient-silver">
//             AUTOMOTIVE EXPERTISE
//           </span>
//         </motion.h1>

//         <motion.p
//           initial={{ opacity: 0, y: 30 }}
//           animate={{ opacity: 1, y: 0 }}
//           transition={{ duration: 1, delay: 0.6, ease: "easeOut" }}
//           className="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto mb-12 font-light"
//         >
//           Intelligent Technical Assistance for Luxury Vehicles
//         </motion.p>

//         {/* Animated car silhouette */}
//         <motion.div
//           initial={{ opacity: 0, scale: 0.8 }}
//           animate={{ opacity: 1, scale: 1 }}
//           transition={{ duration: 1.5, delay: 0.9, ease: "easeOut" }}
//           className="mb-16"
//         >
//           <div className="w-full max-w-2xl mx-auto relative">
//             <div className="absolute inset-0 bg-gradient-silver blur-3xl opacity-30 animate-pulse-glow" />
//             <svg className="w-full h-auto relative z-10" viewBox="0 0 400 200" fill="none" xmlns="http://www.w3.org/2000/svg">
//               <motion.path
//                 d="M 50 150 L 80 150 L 90 130 L 130 130 L 140 110 L 260 110 L 270 130 L 310 130 L 320 150 L 350 150 L 350 160 L 330 160 C 330 170, 320 180, 310 180 C 300 180, 290 170, 290 160 L 110 160 C 110 170, 100 180, 90 180 C 80 180, 70 170, 70 160 L 50 160 Z"
//                 stroke="hsl(0 0% 88%)"
//                 strokeWidth="2"
//                 fill="hsl(0 0% 4%)"
//                 initial={{ pathLength: 0 }}
//                 animate={{ pathLength: 1 }}
//                 transition={{ duration: 2, delay: 1.2, ease: "easeInOut" }}
//               />
//             </svg>
//           </div>
//         </motion.div>
//       </div>

//       {/* Scroll indicator */}
//       <motion.div
//         initial={{ opacity: 0 }}
//         animate={{ opacity: 1 }}
//         transition={{ duration: 1, delay: 2, ease: "easeOut" }}
//         className="absolute bottom-12 left-1/2 transform -translate-x-1/2 cursor-pointer"
//         onClick={handleScroll}
//       >
//         <motion.div
//           animate={{ y: [0, 10, 0] }}
//           transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
//         >
//           <ChevronDown className="w-8 h-8 text-muted-foreground" />
//         </motion.div>
//       </motion.div>
//     </div>
//   );
// };

// export default HeroLanding;
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

const HeroLanding = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const handleScroll = (e: WheelEvent) => {
      if (e.deltaY > 0) {
        navigate("/brand-selection");
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown' || e.key === ' ' || e.key === 'Enter') {
        navigate("/brand-selection");
      }
    };

    window.addEventListener('wheel', handleScroll);
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('wheel', handleScroll);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [navigate]);

  const handleClick = () => {
    navigate("/brand-selection");
  };

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden cursor-pointer" onClick={handleClick}>
      {/* Dynamic background with multiple layers */}
      <div className="absolute inset-0">
        {/* Base gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-zinc-950 via-black to-zinc-900" />
        
        {/* Animated light beams */}
        <motion.div
          animate={{
            opacity: [0.1, 0.3, 0.1],
            scale: [1, 1.5, 1],
          }}
          transition={{
            duration: 15,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          className="absolute -top-1/2 -right-1/2 w-full h-full bg-gradient-radial from-blue-500/20 via-transparent to-transparent blur-3xl"
        />
        
        <motion.div
          animate={{
            opacity: [0.15, 0.35, 0.15],
            scale: [1.5, 1, 1.5],
          }}
          transition={{
            duration: 12,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 2
          }}
          className="absolute -bottom-1/2 -left-1/2 w-full h-full bg-gradient-radial from-purple-500/20 via-transparent to-transparent blur-3xl"
        />

        {/* Subtle scan lines effect */}
        <div 
          className="absolute inset-0 opacity-[0.02]"
          style={{
            backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.03) 2px, rgba(255,255,255,0.03) 4px)'
          }}
        />
      </div>

      {/* Floating particles */}
      {[...Array(20)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-1 h-1 bg-white rounded-full"
          initial={{
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            opacity: Math.random() * 0.5
          }}
          animate={{
            y: [null, Math.random() * window.innerHeight],
            opacity: [null, 0, Math.random() * 0.5],
          }}
          transition={{
            duration: Math.random() * 10 + 10,
            repeat: Infinity,
            ease: "linear",
            delay: Math.random() * 5
          }}
        />
      ))}

      {/* Minimalist logo top left */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 1.5, delay: 0.5 }}
        className="absolute top-8 left-8 z-20"
      >
        <div className="w-12 h-12 border border-white/20 flex items-center justify-center">
          <div className="text-sm font-light tracking-wider">A</div>
        </div>
      </motion.div>

      {/* Main content - centered */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-8">
        {/* Animated lines before text */}
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: "100px" }}
          transition={{ duration: 1.5, delay: 1 }}
          className="h-[1px] bg-gradient-to-r from-transparent via-white to-transparent mb-12"
        />

        {/* Main heading with staggered reveal */}
        <div className="text-center max-w-5xl">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 2, delay: 1.5 }}
          >
            <motion.h1 
              className="text-8xl md:text-9xl font-extralight tracking-[-0.02em] leading-none mb-8"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1.2, delay: 1.8 }}
            >
              AUTOMOTIVE
            </motion.h1>
            
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1.2, delay: 2.2 }}
              className="relative inline-block"
            >
              <h1 className="text-8xl md:text-9xl font-extralight tracking-[-0.02em] leading-none">
                INTELLIGENCE
              </h1>
              {/* Glowing underline */}
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: "100%" }}
                transition={{ duration: 1.5, delay: 3 }}
                className="absolute bottom-0 left-0 h-[2px] bg-gradient-to-r from-transparent via-white to-transparent"
              />
            </motion.div>
          </motion.div>

          {/* Subtitle with fade in */}
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.5 }}
            transition={{ duration: 1.5, delay: 3.5 }}
            className="text-lg md:text-xl font-light tracking-[0.3em] mt-16 uppercase"
          >
            Technical Excellence
          </motion.p>
        </div>

        {/* Animated lines after text */}
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: "100px" }}
          transition={{ duration: 1.5, delay: 4 }}
          className="h-[1px] bg-gradient-to-r from-transparent via-white to-transparent mt-12"
        />
      </div>

      {/* Bottom year/tagline */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.3 }}
        transition={{ duration: 2, delay: 4.5 }}
        className="absolute bottom-8 left-0 right-0 text-center z-20"
      >
        <div className="text-xs tracking-[0.3em] font-light">
          EST. 2025
        </div>
      </motion.div>

      {/* Scroll hint - subtle pulse */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: [0, 0.4, 0] }}
        transition={{ duration: 3, delay: 5, repeat: Infinity }}
        className="absolute bottom-20 left-1/2 -translate-x-1/2 z-20"
      >
        <div className="text-xs tracking-widest">ENTER</div>
      </motion.div>
    </div>
  );
};

export default HeroLanding;
