// import { useNavigate } from "react-router-dom";
// import { ArrowLeft, Car } from "lucide-react";
// import { motion } from "framer-motion";
// import { Button } from "@/components/ui/button";
// import { Card } from "@/components/ui/card";

// interface Brand {
//   id: string;
//   name: string;
//   tagline: string;
//   theme: string;
// }

// /*const brands: Brand[] = [
//   { id: "mercedes", name: "Mercedes-Benz", tagline: "The Best or Nothing", theme: "silver" },
//   { id: "bmw", name: "BMW", tagline: "The Ultimate Driving Machine", theme: "chrome" },
//   { id: "audi", name: "Audi", tagline: "Vorsprung durch Technik", theme: "platinum" },
//   { id: "porsche", name: "Porsche", tagline: "There is No Substitute", theme: "silver" },
//   { id: "lexus", name: "Lexus", tagline: "Experience Amazing", theme: "chrome" },
//   { id: "tesla", name: "Tesla", tagline: "Accelerating Sustainable Transport", theme: "silver" },
//   { id: "jaguar", name: "Jaguar", tagline: "Grace, Space, Pace", theme: "platinum" },
//   { id: "volvo", name: "Volvo", tagline: "Made by Sweden", theme: "chrome" },
//   { id: "genesis", name: "Genesis", tagline: "Designed to Inspire", theme: "silver" },
// ];*/
// const brands: Brand[] = [
//   { id: "toyota", name: "Toyota", tagline: "Let's Go Places", theme: "silver" },
//   { id: "maruti", name: "Maruti Suzuki", tagline: "Way of Life", theme: "chrome" },
//   { id: "hyundai", name: "Hyundai", tagline: "New Thinking. New Possibilities.", theme: "platinum" },
//   { id: "tata", name: "Tata Motors", tagline: "Connecting Aspirations", theme: "silver" },
//   { id: "nissan", name: "Nissan", tagline: "Innovation That Excites", theme: "chrome" },
// ];


// const BrandSelection = () => {
//   const navigate = useNavigate();

//   const cardVariants = {
//     hidden: { opacity: 0, scale: 0.9 },
//     visible: (i: number) => ({
//       opacity: 1,
//       scale: 1,
//       transition: {
//         delay: i * 0.08,
//         duration: 0.6,
//         ease: [0.16, 1, 0.3, 1] as const
//       }
//     })
//   };

//   const handleBrandSelect = (brandId: string) => {
//     navigate(`/model-selection?brand=${brandId}`);
//   };

//   return (
//     <div className="min-h-screen bg-gradient-hero relative overflow-hidden">
//       {/* Animated background particles */}
//       <div className="absolute inset-0 opacity-30">
//         <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-secondary/20 rounded-full blur-3xl animate-float" />
//         <div className="absolute bottom-1/3 right-1/4 w-96 h-96 bg-chrome/20 rounded-full blur-3xl animate-float" style={{ animationDelay: "2s" }} />
//       </div>

//       {/* Header */}
//       <header className="border-b border-border glass-card relative z-10">
//         <div className="container mx-auto px-4 py-6">
//           <div className="flex items-center justify-between">
//             <Button
//               variant="ghost"
//               onClick={() => navigate("/")}
//               className="gap-2 hover:bg-secondary/10"
//             >
//               <ArrowLeft className="w-4 h-4" />
//               Back
//             </Button>
//             <div className="text-sm text-muted-foreground">
//               Home <span className="text-foreground mx-2">›</span> Brand Selection
//             </div>
//           </div>
//         </div>
//       </header>

//       {/* Main Content */}
//       <div className="container mx-auto px-4 py-16 relative z-10">
//         {/* Title Section */}
//         <motion.div
//           initial={{ opacity: 0, y: -30 }}
//           animate={{ opacity: 1, y: 0 }}
//           transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] as const }}
//           className="text-center mb-16"
//         >
//           <div className="inline-flex items-center gap-2 px-6 py-3 rounded-full glass-card mb-6">
//             <Car className="w-5 h-5 text-secondary" />
//             <span className="text-sm font-medium">Premium Automotive Brands</span>
//           </div>
          
//           <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
//             <span className="text-gradient-silver">
//               Select Your
//             </span>
//             <br />
//             <span className="text-foreground">
//               Manufacturer
//             </span>
//           </h1>
          
//           <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
//             Choose from our collection of world-class automotive brands
//           </p>
//         </motion.div>

//         {/* Brand Grid */}
//         <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
//           {brands.map((brand, index) => (
//             <motion.div
//               key={brand.id}
//               custom={index}
//               variants={cardVariants}
//               initial="hidden"
//               animate="visible"
//               whileHover={{ 
//                 y: -25, 
//                 rotateY: 10,
//                 scale: 1.05,
//                 transition: { duration: 0.5, ease: [0.4, 0, 0.2, 1] as const }
//               }}
//               onClick={() => handleBrandSelect(brand.id)}
//               className="cursor-pointer perspective-1000"
//               style={{ perspective: "1000px" }}
//             >
//               <Card className="relative overflow-hidden h-full p-10 glass-card border-2 border-transparent hover:border-secondary/50 transition-all duration-500 group">
//                 {/* Animated gradient background on hover */}
//                 <div className="absolute inset-0 bg-gradient-silver opacity-0 group-hover:opacity-10 transition-opacity duration-500" />
                
//                 {/* Glow effect */}
//                 <div className="absolute inset-0 shadow-glow-silver opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                
//                 {/* Animated border gradient */}
//                 <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
//                   <div className="absolute inset-0 border-2 border-transparent bg-gradient-to-r from-transparent via-secondary to-transparent bg-clip-border animate-pulse-glow" />
//                 </div>
                
//                 <div className="relative z-10 text-center">
//                   {/* Brand Icon */}
//                   <div className="w-24 h-24 rounded-2xl bg-gradient-silver flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform duration-500 shadow-elegant">
//                     <Car className="w-12 h-12 text-background" />
//                   </div>
                  
//                   {/* Brand Name */}
//                   <h3 className="text-3xl font-bold mb-3 text-gradient-silver group-hover:scale-105 transition-transform duration-500">
//                     {brand.name}
//                   </h3>
                  
//                   {/* Divider */}
//                   <div className="w-16 h-0.5 bg-gradient-silver mx-auto mb-4 group-hover:w-24 transition-all duration-500" />
                  
//                   {/* Tagline */}
//                   <p className="text-muted-foreground text-sm italic leading-relaxed">
//                     "{brand.tagline}"
//                   </p>
                  
//                   {/* Hover Indicator */}
//                   <div className="mt-6 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
//                     <div className="text-xs uppercase tracking-wider text-secondary font-semibold">
//                       Select Brand →
//                     </div>
//                   </div>
//                 </div>
//               </Card>
//             </motion.div>
//           ))}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default BrandSelection;

import { useNavigate } from "react-router-dom";
import { ArrowLeft, Car } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface Brand {
  id: string;
  name: string;
  tagline: string;
  theme: string;
}

const brands: Brand[] = [
  { id: "toyota", name: "Toyota", tagline: "Let's Go Places", theme: "silver" },
  { id: "maruti", name: "Maruti Suzuki", tagline: "Way of Life", theme: "chrome" },
  { id: "hyundai", name: "Hyundai", tagline: "New Thinking. New Possibilities.", theme: "platinum" },
  { id: "tata", name: "Tata Motors", tagline: "Connecting Aspirations", theme: "silver" },
  { id: "nissan", name: "Nissan", tagline: "Innovation That Excites", theme: "chrome" },
];

const BrandSelection = () => {
  const navigate = useNavigate();

  const cardVariants = {
    hidden: { opacity: 0, scale: 0.9 },
    visible: (i: number) => ({
      opacity: 1,
      scale: 1,
      transition: {
        delay: i * 0.08,
        duration: 0.6,
        ease: [0.16, 1, 0.3, 1] as const
      }
    })
  };

  const handleBrandSelect = (brandId: string) => {
    console.log("🚗 Brand selected:", brandId);
    navigate(`/model-selection?brand=${brandId}`);
  };

  return (
    <div className="min-h-screen bg-gradient-hero relative overflow-hidden">
      {/* Animated background particles */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-secondary/20 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-1/3 right-1/4 w-96 h-96 bg-chrome/20 rounded-full blur-3xl animate-float" style={{ animationDelay: "2s" }} />
      </div>

      {/* Header */}
      <header className="border-b border-border glass-card relative z-10">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <Button
              variant="ghost"
              onClick={() => navigate("/")}
              className="gap-2 hover:bg-secondary/10"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </Button>
            <div className="text-sm text-muted-foreground">
              Home <span className="text-foreground mx-2">›</span> Brand Selection
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-16 relative z-10">
        {/* Title Section */}
        <motion.div
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] as const }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 px-6 py-3 rounded-full glass-card mb-6">
            <Car className="w-5 h-5 text-secondary" />
            <span className="text-sm font-medium">Premium Automotive Brands</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
            <span className="text-gradient-silver">
              Select Your
            </span>
            <br />
            <span className="text-foreground">
              Manufacturer
            </span>
          </h1>
          
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Choose from our collection of world-class automotive brands
          </p>
        </motion.div>

        {/* Brand Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
          {brands.map((brand, index) => (
            <motion.div
              key={brand.id}
              custom={index}
              variants={cardVariants}
              initial="hidden"
              animate="visible"
              whileHover={{ 
                y: -25, 
                rotateY: 10,
                scale: 1.05,
                transition: { duration: 0.5, ease: [0.4, 0, 0.2, 1] as const }
              }}
              onClick={() => handleBrandSelect(brand.id)}
              className="cursor-pointer perspective-1000"
              style={{ perspective: "1000px" }}
            >
              <Card className="relative overflow-hidden h-full p-10 glass-card border-2 border-transparent hover:border-secondary/50 transition-all duration-500 group">
                {/* Animated gradient background on hover */}
                <div className="absolute inset-0 bg-gradient-silver opacity-0 group-hover:opacity-10 transition-opacity duration-500" />
                
                {/* Glow effect */}
                <div className="absolute inset-0 shadow-glow-silver opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                
                {/* Animated border gradient */}
                <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
                  <div className="absolute inset-0 border-2 border-transparent bg-gradient-to-r from-transparent via-secondary to-transparent bg-clip-border animate-pulse-glow" />
                </div>
                
                <div className="relative z-10 text-center">
                  {/* Brand Icon */}
                  <div className="w-24 h-24 rounded-2xl bg-gradient-silver flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform duration-500 shadow-elegant">
                    <Car className="w-12 h-12 text-background" />
                  </div>
                  
                  {/* Brand Name */}
                  <h3 className="text-3xl font-bold mb-3 text-gradient-silver group-hover:scale-105 transition-transform duration-500">
                    {brand.name}
                  </h3>
                  
                  {/* Divider */}
                  <div className="w-16 h-0.5 bg-gradient-silver mx-auto mb-4 group-hover:w-24 transition-all duration-500" />
                  
                  {/* Tagline */}
                  <p className="text-muted-foreground text-sm italic leading-relaxed">
                    "{brand.tagline}"
                  </p>
                  
                  {/* Hover Indicator */}
                  <div className="mt-6 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
                    <div className="text-xs uppercase tracking-wider text-secondary font-semibold">
                      Select Brand →
                    </div>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default BrandSelection;