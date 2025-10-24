export interface Property {
  id: string;
  title: string;
  price: number;
  location: string;
  bedrooms: number;
  bathrooms: number;
  sqft: number;
  image: string;
  type: string;
  features: string[];
  keywords: string[];
}

export const mockProperties: Property[] = [
  {
    id: "1",
    title: "Modern Family Home",
    price: 650000,
    location: "Oakville, ON",
    bedrooms: 4,
    bathrooms: 3,
    sqft: 2800,
    image: "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBob3VzZSUyMGV4dGVyaW9yfGVufDF8fHx8MTc1Nzg0OTU1Mnww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    type: "House",
    features: ["Garage", "Garden", "Updated Kitchen", "Hardwood Floors", "Walk-in Closet"],
    keywords: ["family", "modern", "house", "garage", "garden", "suburban", "spacious", "4 bedroom", "oakville"]
  },
  {
    id: "2",
    title: "Luxury Downtown Condo",
    price: 850000,
    location: "Downtown Toronto, ON",
    bedrooms: 2,
    bathrooms: 2,
    sqft: 1200,
    image: "https://images.unsplash.com/photo-1638454668466-e8dbd5462f20?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxsdXh1cnklMjBhcGFydG1lbnQlMjBpbnRlcmlvcnxlbnwxfHx8fDE3NTc5MDM3NjZ8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    type: "Condo",
    features: ["City View", "Concierge", "Gym", "Pool", "Balcony"],
    keywords: ["luxury", "condo", "downtown", "city", "apartment", "high rise", "amenities", "2 bedroom", "toronto"]
  },
  {
    id: "3",
    title: "Charming Suburban Home",
    price: 425000,
    location: "Mississauga, ON",
    bedrooms: 3,
    bathrooms: 2,
    sqft: 1850,
    image: "https://images.unsplash.com/photo-1570129477492-45c003edd2be?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmYW1pbHklMjBob21lJTIwc3VidXJiYW58ZW58MXx8fHwxNzU3ODUzODc4fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    type: "House",
    features: ["Fireplace", "Deck", "Finished Basement", "Driveway"],
    keywords: ["suburban", "family", "house", "basement", "deck", "affordable", "3 bedroom", "mississauga"]
  },
  {
    id: "4",
    title: "Executive Townhouse",
    price: 780000,
    location: "Markham, ON",
    bedrooms: 3,
    bathrooms: 3,
    sqft: 2200,
    image: "https://images.unsplash.com/photo-1659621222272-f65c27b6f182?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxkb3dudG93biUyMGNvbmRvJTIwY2l0eXxlbnwxfHx8fDE3NTc5MDM3Njd8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    type: "Townhouse",
    features: ["Garage", "Ensuite", "Open Concept", "Granite Counters"],
    keywords: ["townhouse", "executive", "modern", "open concept", "garage", "3 bedroom", "markham"]
  },
  {
    id: "5",
    title: "Starter Condo",
    price: 320000,
    location: "North York, ON",
    bedrooms: 1,
    bathrooms: 1,
    sqft: 650,
    image: "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxsdXh1cnklMjBob3VzZSUyMGV4dGVyaW9yfGVufDF8fHx8MTc1Nzg0OTU1Mnww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    type: "Condo",
    features: ["Transit Access", "Storage", "Laundry", "Balcony"],
    keywords: ["starter", "condo", "affordable", "transit", "1 bedroom", "first time buyer", "north york"]
  },
  {
    id: "6",
    title: "Luxury Estate",
    price: 1200000,
    location: "Richmond Hill, ON",
    bedrooms: 5,
    bathrooms: 4,
    sqft: 4200,
    image: "https://images.unsplash.com/photo-1570129477492-45c003edd2be?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmYW1pbHklMjBob21lJTIwc3VidXJiYW58ZW58MXx8fHwxNzU3ODUzODc4fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    type: "House",
    features: ["Pool", "Wine Cellar", "3-Car Garage", "Home Theatre", "Chef's Kitchen"],
    keywords: ["luxury", "estate", "pool", "large", "expensive", "5 bedroom", "richmond hill"]
  }
];

export function findMatchingProperties(userMessage: string, maxResults: number = 3): Property[] {
  const message = userMessage.toLowerCase();
  
  // Extract key information from user message
  const bedroomMatch = message.match(/(\d+)\s*(?:bed|bedroom)/);
  const budgetMatch = message.match(/(?:under|below|max|maximum|budget)\s*(?:of)?\s*\$?(\d+(?:,\d{3})*(?:k|000)?)/);
  const typeMatch = message.match(/\b(condo|house|townhouse|apartment)\b/);
  
  const preferences = {
    bedrooms: bedroomMatch ? parseInt(bedroomMatch[1]) : null,
    maxBudget: budgetMatch ? parseBudget(budgetMatch[1]) : null,
    type: typeMatch ? typeMatch[1] : null
  };

  // Score properties based on how well they match user preferences
  const scoredProperties = mockProperties.map(property => {
    let score = 0;
    
    // Bedroom match
    if (preferences.bedrooms && property.bedrooms === preferences.bedrooms) {
      score += 50;
    } else if (preferences.bedrooms && Math.abs(property.bedrooms - preferences.bedrooms) === 1) {
      score += 25;
    }
    
    // Budget match
    if (preferences.maxBudget && property.price <= preferences.maxBudget) {
      score += 40;
    } else if (preferences.maxBudget && property.price > preferences.maxBudget) {
      score -= 30;
    }
    
    // Type match
    if (preferences.type && property.type.toLowerCase() === preferences.type) {
      score += 30;
    }
    
    // Keyword matching
    property.keywords.forEach(keyword => {
      if (message.includes(keyword)) {
        score += 10;
      }
    });
    
    // Location matching
    if (message.includes(property.location.toLowerCase()) || 
        property.location.toLowerCase().includes(message.split(' ').find(word => word.length > 3) || '')) {
      score += 20;
    }
    
    return { property, score };
  });

  // Sort by score and return top matches
  return scoredProperties
    .sort((a, b) => b.score - a.score)
    .slice(0, maxResults)
    .map(item => item.property);
}

function parseBudget(budgetStr: string): number {
  const cleanStr = budgetStr.replace(/,/g, '');
  if (cleanStr.endsWith('k')) {
    return parseInt(cleanStr.slice(0, -1)) * 1000;
  } else if (cleanStr.endsWith('000')) {
    return parseInt(cleanStr);
  } else {
    return parseInt(cleanStr);
  }
}