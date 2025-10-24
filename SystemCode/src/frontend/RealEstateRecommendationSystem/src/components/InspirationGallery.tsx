import { Card, CardContent } from "./ui/card";
import { Badge } from "./ui/badge";
import { ImageWithFallback } from './figma/ImageWithFallback';

interface InspirationImage {
  id: string;
  title: string;
  image: string;
  description: string;
}

interface InspirationGalleryProps {
  images: InspirationImage[];
  loading?: boolean;
}

export function InspirationGallery({ images, loading = false }: InspirationGalleryProps) {
  if (loading) {
    return (
      <div className="space-y-4 p-4">
        <div className="animate-pulse">
          <div className="h-4 bg-muted rounded w-1/3 mb-4"></div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="space-y-3">
                <div className="h-48 bg-muted rounded"></div>
                <div className="h-4 bg-muted rounded w-3/4"></div>
                <div className="h-3 bg-muted rounded w-1/2"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (images.length === 0) {
    return null;
  }

  return (
    <div className="p-4 border-t bg-muted/30">
      <h3 className="font-semibold mb-4 text-lg">Design Inspiration</h3>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {images.map((item) => (
          <Card key={item.id} className="overflow-hidden hover:shadow-lg transition-shadow">
            <div className="relative">
              <ImageWithFallback
                src={item.image}
                alt={item.title}
                className="w-full h-48 object-cover"
              />
              <Badge className="absolute top-3 left-3 bg-white text-gray-800">
                Inspiration
              </Badge>
            </div>
            
            <CardContent className="p-4">
              <h4 className="font-semibold text-base mb-2">{item.title}</h4>
              <p className="text-sm text-muted-foreground">{item.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}